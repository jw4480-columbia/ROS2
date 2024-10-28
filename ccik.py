#!/usr/bin/env python3

import math
import numpy
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from custom_msg.msg import CartesianCommand
from urdf_parser_py.urdf import URDF
import random
import transforms3d
import transforms3d._gohlketransforms as tf
from threading import Thread, Lock


'''This is a class which will perform both cartesian control and inverse
   kinematics'''
class CCIK(Node):
    def __init__(self):
        super().__init__('ccik')
    #Load robot from parameter server
        # self.robot = URDF.from_parameter_server()
        self.declare_parameter(
            'rd_file', rclpy.Parameter.Type.STRING)
        robot_desription = self.get_parameter('rd_file').value
        with open(robot_desription, 'r') as file:
            robot_desription_text = file.read()
        # print(robot_desription_text)
        self.robot = URDF.from_xml_string(robot_desription_text)

    #Subscribe to current joint state of the robot
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.get_joint_state, 10)

    #This will load information about the joints of the robot
        self.num_joints = 0
        self.joint_names = []
        self.q_current = []
        self.joint_axes = []
        self.get_joint_info()

    #This is a mutex
        self.mutex = Lock()

    #Subscribers and publishers for for cartesian control
        self.cartesian_command_sub = self.create_subscription(
            CartesianCommand, '/cartesian_command', self.get_cartesian_command, 10)
        self.velocity_pub = self.create_publisher(JointState, '/joint_velocities', 10)
        self.joint_velocity_msg = JointState()

    #Subscribers and publishers for numerical IK
        self.ik_command_sub = self.create_subscription(
            Transform, '/ik_command', self.get_ik_command, 10)
        self.joint_command_pub = self.create_publisher(JointState, '/joint_command', 10)
        self.joint_command_msg = JointState()

    '''This is a function which will collect information about the robot which
       has been loaded from the parameter server. It will populate the variables
       self.num_joints (the number of joints), self.joint_names and
       self.joint_axes (the axes around which the joints rotate)'''
    def get_joint_info(self):
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link
        #self.get_logger().info(f'Joint names {self.joint_names}')
        #self.get_logger().info(f'Joint axes {self.joint_axes}')



    '''This is the callback which will be executed when the cartesian control
       recieves a new command. The command will contain information about the
       secondary objective and the target q0. At the end of this callback, 
       you should publish to the /joint_velocities topic.'''
    def get_cartesian_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR CARTESIAN CONTROL HERE
        # compute transform from current end-effector pose to desired one

        #set the gain current
        g_q=1
        g_sec=3
        
        #compute Jacobian and its psuedoinverse
        joint_transforms, b_T_ee = self.forward_kinematics(self.q_current)

        J_current=self.get_jacobian(b_T_ee,joint_transforms)

        J_s_p=numpy.linalg.pinv(J_current,0.01)

        # compute transform from current end-effector pose to desired one
        b_Trans_des=command.x_target
        b_translation_des=b_Trans_des.translation

        b_t_des=[b_translation_des.x,b_translation_des.y,b_translation_des.z]
        Q_des=b_Trans_des.rotation
        q_x=Q_des.x
        q_y=Q_des.y
        q_z=Q_des.z
        q_w=Q_des.w
        quat_des=[q_w,q_x,q_y,q_z]
        b_R_des=transforms3d.quaternions.quat2mat(quat_des)
        b_T_des=transforms3d.affines.compose(b_t_des,b_R_des,(1,1,1))

        b_t_ee,b_R_ee,Z,S=transforms3d.affines.decompose(b_T_ee)
        ee_R_b=numpy.transpose(b_R_ee)
        ee_t_b=numpy.matmul(-1*ee_R_b,b_t_ee)
        ee_T_b=numpy.linalg.inv(b_T_ee)

        ee_T_des=numpy.dot(ee_T_b,b_T_des)

        ee_t_des,ee_R_des,Z,S=transforms3d.affines.decompose(ee_T_des)
        axis,angle=self.rotation_from_matrix(ee_R_des)
        ee_r_des=axis*angle
        x_ee=numpy.concatenate((ee_t_des,ee_r_des),axis=0)

        v_ee=g_q*x_ee
        
        if numpy.linalg.norm(v_ee[:3])>0.1:
         for i in range (0,3):
             v_ee[i]=0.1*v_ee[i]/numpy.linalg.norm(v_ee[:3]) #divide by norm to get the condition
        if numpy.linalg.norm(v_ee[3:6])>1:
         for i in range (3,6) :
             v_ee[i]=v_ee[i]/numpy.linalg.norm(v_ee[3:6])

        q_d_des=numpy.dot(J_s_p,v_ee)
        v_veri=numpy.matmul(J_current,q_d_des)
        if numpy.linalg.norm(v_ee[3:6])>1:
         for i in range (3,6) :
             q_d_des[i]=q_d_des[i]/numpy.linalg.norm(q_d_des[3:6]) 

        J_p=numpy.linalg.pinv(J_current)
        if(command.secondary_objective):
            r_0=command.q0_target
            q_0=self.q_current[0]
            q_d_sec=numpy.zeros(self.num_joints)
            q_d_sec[0]=g_sec*(r_0-q_0)
            I=numpy.eye(self.num_joints)
            q_d_null=numpy.dot((I-numpy.dot(J_p,J_current)),q_d_sec)
            q_d_des=q_d_des+q_d_null

        velocity=[]
        for i in range(self.num_joints):
            velocity.append(q_d_des[i])
           
        self.joint_velocity_msg.name=self.joint_names
        self.joint_velocity_msg.velocity=velocity
        self.velocity_pub.publish(self.joint_velocity_msg)
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This is a function which will assemble the jacobian of the robot using the
       current joint transforms and the transform from the base to the end
       effector (b_T_ee). Both the cartesian control callback and the
       inverse kinematics callback will make use of this function.
       Usage: J = self.get_jacobian(b_T_ee, joint_transforms)'''
    def get_jacobian(self, b_T_ee, joint_transforms):
        J = numpy.zeros((6,self.num_joints))
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR ASSEMBLING THE CURRENT JACOBIAN HERE
        for i in range(self.num_joints):

            b_T_j=joint_transforms[i]
            j_T_b=numpy.linalg.inv(b_T_j)
            j_T_ee=numpy.dot(j_T_b,b_T_ee)
            j_t_ee,j_R_ee,Z,S=transforms3d.affines.decompose(j_T_ee)

            j_s_ee=skew_mat(j_t_ee)
            ee_R_j=numpy.transpose(j_R_ee)
            A=numpy.dot(-1*ee_R_j,j_s_ee)
            Vi=numpy.append(A,ee_R_j,axis=0)

            axis=numpy.transpose(self.joint_axes[i])
            J[:,i]=numpy.dot(Vi,axis)



        #--------------------------------------------------------------------------
        return J

    '''This is the callback which will be executed when the inverse kinematics
       recieve a new command. The command will contain information about desired
       end effector pose relative to the root of your robot. At the end of this
       callback, you should publish to the /joint_command topic. This should not
       search for a solution indefinitely - there should be a time limit. When
       searching for two matrices which are the same, we expect numerical
       precision of 10e-3.'''
    def get_ik_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR INVERSE KINEMATICS HERE
        print('Solution starts here:')
        print(time.clock_gettime(time.CLOCK_REALTIME))
        count=0
        q_c=numpy.random.uniform(low=0.0,high=2*math.pi,size=self.num_joints)
        
        b_Trans_des=command
        b_translation_des=b_Trans_des.translation

        b_t_des=[b_translation_des.x,b_translation_des.y,b_translation_des.z]
        Q_des=b_Trans_des.rotation
        q_x=Q_des.x
        q_y=Q_des.y
        q_z=Q_des.z
        q_w=Q_des.w
        quat_des=[q_w,q_x,q_y,q_z]
        b_R_des=transforms3d.quaternions.quat2mat(quat_des)
        b_T_des=transforms3d.affines.compose(b_t_des,b_R_des,(1,1,1))

        T_start=time.clock_gettime(time.CLOCK_REALTIME)
        while count<3:
            T_now=time.clock_gettime(time.CLOCK_REALTIME)
            num_sat=0
            joint_transforms,b_T_c=self.forward_kinematics(q_c)
            c_T_b=numpy.linalg.inv(b_T_c)	

            c_T_des=numpy.dot(c_T_b,b_T_des)
            c_t_des,c_R_des,Z,S=transforms3d.affines.decompose(c_T_des)
            axis,angle=self.rotation_from_matrix(c_R_des)
            c_r_des=axis*angle
	       
            dx=numpy.concatenate((c_t_des,c_r_des),axis=0)
	    
            J=self.get_jacobian(b_T_c, joint_transforms)
            Jp=numpy.linalg.pinv(J)
            
            d_q=numpy.dot(Jp,dx)
            
            q_c = q_c+d_q
            
            for i in range(0,self.num_joints):            
                if abs(d_q[i])<0.001:
                    num_sat=num_sat+1
                else:
                    break
            #print(num_sat)
            if num_sat==self.num_joints:
                break
            else:
                T_p=(T_now-T_start)
                if T_p>3:
                    count+=1
                    print(T_p)
                    print('failed attempt number:')
                    print(count)
                    
                    T_start=time.clock_gettime(time.CLOCK_REALTIME)
                continue


        if count==3:
            q_c=self.q_current
            print('failed to solve IK')


        position=[]
        for i in range(self.num_joints):
            position.append(q_c[i])
        
        self.joint_command_msg.name=self.joint_names
        self.joint_command_msg.position=position
        self.joint_command_pub.publish(self.joint_command_msg)


        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This function will return the angle-axis representation of the rotation
       contained in the input matrix. Use like this: 
       angle, axis = rotation_from_matrix(R)'''
    def rotation_from_matrix(self, matrix):
        R = numpy.array(matrix, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = numpy.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = numpy.linalg.eig(R)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis

    '''This is the function which will perform forward kinematics for your 
       cartesian control and inverse kinematics functions. It takes as input
       joint values for the robot and will return an array of 4x4 transforms
       from the base to each link of the robot, as well as the transform from
       the base to the end effector.
       Usage: joint_transforms, b_T_ee = self.forward_kinematics(joint_values)'''
    def forward_kinematics(self, joint_values):
        joint_transforms = []

        link = self.robot.get_root()
        T = tf.identity_matrix()

        while True:
            if link not in self.robot.child_map:
                break

            (joint_name, next_link) = self.robot.child_map[link][0]
            joint = self.robot.joint_map[joint_name]

            T_l = numpy.dot(tf.translation_matrix(joint.origin.xyz), tf.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2], 'rxyz'))
            T = numpy.dot(T, T_l)
            '''print('link transform:')
            print(joint_name)
            print(T_l)'''

            if joint.type != "fixed":
                joint_transforms.append(T)
                q_index = self.joint_names.index(joint_name)
                T_j = tf.rotation_matrix(joint_values[q_index], numpy.asarray(joint.axis))
                T = numpy.dot(T, T_j)

            link = next_link
        return joint_transforms, T #where T = b_T_ee

    '''This is the callback which will recieve and store the current robot
       joint states.'''
    def get_joint_state(self, msg):
        self.mutex.acquire()
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])
        self.mutex.release()

def skew_mat(M):
	skew = numpy.zeros((3,3))
	skew[0,1] = -M[2]
	skew[0,2] = M[1]
	skew[1,0] = M[2]
	skew[1,2] = -M[0]
	skew[2,0] = -M[1]
	skew[2,1] = M[0]
	return skew


def main(args = None):
    rclpy.init()
    ccik = CCIK()
    rclpy.spin(ccik)
    ccik.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
