#!/usr/bin/env python3
import numpy
import random
import sys

import moveit_msgs.msg
import moveit_msgs.srv
import rclpy
from rclpy.node import Node
import rclpy.duration
import transforms3d._gohlketransforms as tf
import transforms3d
from urdf_parser_py.urdf import URDF
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform, PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import copy
import math

def convert_to_message(T):
    t = Pose()
    position, Rot, _, _ = transforms3d.affines.decompose(T)
    orientation = transforms3d.quaternions.mat2quat(Rot)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[1]
    t.orientation.y = orientation[2]
    t.orientation.z = orientation[3]
    t.orientation.w = orientation[0]        
    return t

def convert_to_transform(M):
    trans=[M.translation.x,M.translation.y,M.translation.z]
    Q_des=M.rotation
    q_x=Q_des.x
    q_y=Q_des.y
    q_z=Q_des.z
    q_w=Q_des.w
    quat_des=[q_w,q_x,q_y,q_z]
    R=transforms3d.quaternions.quat2mat(quat_des)
    T=transforms3d.affines.compose(trans,R,(1,1,1))

    return T

def to_vector(q1,q2):
    return numpy.subtract(q1,q2)

def to_unit_vector(v):
    return v/numpy.linalg.norm(v)

def get_length(v):
    return numpy.linalg.norm(v)

def is_vector_iden(v1,v2):
    bool = True
    if len(v1)!=len(v2):
        bool = False
    else:
        for i in range(len(v1)):
            if v1[i]!=v2[i]:
                bool = False
                break
    return bool

class MoveArm(Node):
    def __init__(self):
        super().__init__('move_arm')

        #Loads the robot model, which contains the robot's kinematics information
        self.ee_goal = None
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        #Loads the robot model, which contains the robot's kinematics information
        self.declare_parameter(
            'rd_file', rclpy.Parameter.Type.STRING)
        robot_desription = self.get_parameter('rd_file').value
        with open(robot_desription, 'r') as file:
            robot_desription_text = file.read()
        self.robot = URDF.from_xml_string(robot_desription_text)
        self.base = self.robot.get_root()
        self.get_joint_info()


        self.service_cb_group1 = MutuallyExclusiveCallbackGroup()
        self.service_cb_group2 = MutuallyExclusiveCallbackGroup()
        self.q_current = []

        # Wait for moveit IK service
        self.ik_service = self.create_client(moveit_msgs.srv.GetPositionIK, '/compute_ik', callback_group=self.service_cb_group1)
        while not self.ik_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')
        self.get_logger().info('IK service ready')

        # Wait for validity check service
        self.state_valid_service = self.create_client(moveit_msgs.srv.GetStateValidity, '/check_state_validity',
                                                      callback_group=self.service_cb_group2)
        while not self.state_valid_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for state validity service...')
        self.get_logger().info('State validity service ready')

        # MoveIt parameter
        self.group_name = 'arm'
        self.get_logger().info(f'child map: \n{self.robot.child_map}')

        #Subscribe to topics
        self.sub_joint_states = self.create_subscription(JointState, '/joint_states', self.get_joint_state, 10)
        self.goal_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_goal = self.create_subscription(Transform, '/motion_planning_goal', self.motion_planning_cb, 2,
                                                 callback_group=self.goal_cb_group)
        self.current_obstacle = "NONE"
        self.sub_obs = self.create_subscription(String, '/obstacle', self.get_obstacle, 10)

        #Set up publisher
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)

        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(0.1, self.motion_planning_timer, callback_group=self.timer_cb_group)

    
    def get_joint_state(self, msg):
        '''This callback provides you with the current joint positions of the robot 
        in member variable q_current.
        '''
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])

    
    def get_obstacle(self, msg):
        '''This callback provides you with the name of the current obstacle which
        exists in the RVIZ environment. Options are "None", "Simple", "Hard",
        or "Super". '''
        self.current_obstacle = msg.data

    def motion_planning_cb(self, ee_goal):
        self.get_logger().info("Motion planner goal received.")
        if self.ee_goal is not None:
            self.get_logger().info("Motion planner busy. Please try again later.")
            return
        self.ee_goal = ee_goal

    def motion_planning_timer(self):
        if self.ee_goal is not None:
            self.get_logger().info("Calling motion planner")            
            self.motion_planning(self.ee_goal)
            self.ee_goal = None
            self.get_logger().info("Motion planner done")            
                
   
    def get_RRT(self,q_current,q_desired):
        q_current=numpy.array(q_current)
        starter=RRTBranch(q_current,q_current)
        RRT=[]
        RRT.append(starter)

        while True:
            q_rand=[]
            for k in range(self.num_joints):
                RAND=random.uniform(-1*numpy.pi,numpy.pi)
                q_rand.append(RAND)
            
            q_rand=numpy.array(q_rand)

            p_close=self.find_closest_point_in_tree(RRT,q_rand)
            q_close=p_close.q
            v_target=to_vector(q_rand,q_close)
            if numpy.linalg.norm(v_target)>0.5:
                    v_target=to_unit_vector(v_target)*0.5
            q_target=q_close+v_target
            if self.is_collision_free(q_close,q_target)==True:
                self.get_logger().info('a new branch added')
                new_branch=RRTBranch(q_close,q_target)
                RRT.append(new_branch)
                if self.is_collision_free(q_target,q_desired)==True:
                    new_branch=RRTBranch(q_target,q_desired)
                    RRT.append(new_branch)
                    self.get_logger().info('reached goal')
                    break

        #trace back
        self.get_logger().info(str(len(RRT)))
        '''for i in range(len(RRT)):
            self.get_logger().info('point= '+str(RRT[i].q)+'parent point='+str(RRT[i].parent))
        '''
        q_list=[]
        p_desired=RRT[-1]
        q_list.append(q_desired)
        q_track=p_desired.parent
        index=len(RRT)-1

        while index>0:
            index=self.find_track_index(RRT,q_track)
            q_back=RRT[index].parent
            q_list.insert(0,RRT[index].q)
            q_track=q_back
        self.get_logger().info('traceback done')
        self.get_logger().info(str(q_list))

        
        #short cuts
        temp=[]
        temp.append(q_list[0])
        i=-1
        while i <len(q_list)-2:
            i=i+1
            self.get_logger().info('examine point:' + str(i))
            for j in range(len(q_list)-i-1):
                self.get_logger().info('refer to point : '+str(len(q_list)-1-j))
                if self.is_collision_free(q_list[i],q_list[len(q_list)-1-j]):
                    self.get_logger().info('find short cut for examine point:'+str(i)+' at point: '+str(len(q_list)-1-j))
                    temp.append(q_list[len(q_list)-1-j])
                    i=len(q_list)-1-j
                    self.get_logger().info('i updated to: '+str(i))
                    break
        q_list=temp
        self.get_logger().info('shortcut done')
        self.get_logger().info('traj after showt cut: '+ str(q_list))

        #resampling
        q_new=[]
        q_new.append(q_list[0])
        for i in range(len(q_list)-1):
            v=to_vector(q_list[i+1],q_list[i])
            norm_v=numpy.linalg.norm(v)
            num_points=norm_v/0.5
            num_points_ceil=int(numpy.ceil(num_points))
            self.get_logger().info('sampling step size: '+str(num_points_ceil))
            step=numpy.divide(v,num_points_ceil)
            for j in range(num_points_ceil):
                q_new.append(q_list[i]+(j+1)*step)
        self.get_logger().info(str(q_new))
        self.get_logger().info('sampling done')
        return q_new
        
            

    def motion_planning(self, ee_goal: Transform): 
        '''Callback function for /motion_planning_goal. This is where you will
        implement your RRT motion planning which is to generate a joint
        trajectory for your manipulator. You are welcome to add other functions
        to this class (i.e. an is_segment_valid" function will likely come in 
        handy multiple times in the motion planning process and it will be 
        easiest to make this a seperate function and then call it from motion
        planning). You may also create trajectory shortcut and trajectory 
        sample functions if you wish, which will also be called from the 
        motion planning function.

        Args: 
            ee_goal: Transform() object describing the desired base to 
            end-effector transformation 
        '''
        b_T_ee_goal=convert_to_transform(ee_goal)
        q_current=self.q_current
        q_goal=self.IK(b_T_ee_goal)
        '''if len(q_goal==0):
            print('goal invalid')
            return'''
        self.get_logger().info('solved to configuration space:')
        self.get_logger().info(str(numpy.size(q_goal)))

        q_list=self.get_RRT(q_current,q_goal)
        #self.get_logger().info(str(q_list))
        traj=self.to_trajectory(q_list)
        self.pub.publish(traj)
        
        # TODO: implement motion_planning here
        



    def find_closest_point_in_tree(self, tree, r):
        shortest_distance = numpy.linalg.norm(r-tree[0].q)
        closest_point = tree[0]
        for i in range(1, len(tree)-1):
            if shortest_distance > numpy.linalg.norm(r-tree[i].q):
                shortest_distance = numpy.linalg.norm(r-tree[i].q)
                closest_point = tree[i]
        return closest_point


    def find_track_index(self,RRT,parent):
        index=len(RRT)-1
        for i in range(len(RRT)):
            if is_vector_iden(parent,RRT[i].q):
                index=i
                break

        return index


    def IK(self, T_goal):
        """ This function will perform IK for a given transform T of the 
        end-effector. It .

        Returns:
            q: returns a list q[] of values, which are the result 
            positions for the joints of the robot arm, ordered from proximal 
            to distal. If no IK solution is found, it returns an empy list
        """

        req = moveit_msgs.srv.GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state.name = self.joint_names
        req.ik_request.robot_state.joint_state.position = list(numpy.zeros(self.num_joints))
        req.ik_request.robot_state.joint_state.velocity = list(numpy.zeros(self.num_joints))
        req.ik_request.robot_state.joint_state.effort = list(numpy.zeros(self.num_joints))
        req.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = 'base'
        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        
        self.get_logger().info('Sending IK request...')
        res = self.ik_service.call(req)
        self.get_logger().info('IK request returned')
        
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = res.solution.joint_state.position
        for i in range(0,len(q)):
            while (q[i] < -math.pi): 
                q[i] = q[i] + 2 * math.pi
            while (q[i] > math.pi): 
                q[i] = q[i] - 2 * math.pi
        return q

    
    def get_joint_info(self):
        '''This is a function which will collect information about the robot which
        has been loaded from the parameter server. It will populate the variables
        self.num_joints (the number of joints), self.joint_names and
        self.joint_axes (the axes around which the joints rotate)
        '''
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
        self.get_logger().info('Num joints: %d' % (self.num_joints))


    
    def is_state_valid(self, q):
        """ This function checks if a set of joint angles q[] creates a valid state,
        or one that is free of collisions. The values in q[] are assumed to be values
        for the joints of the UR5 arm, ordered from proximal to distal.

        Returns:
            bool: true if state is valid, false otherwise
        """
        req = moveit_msgs.srv.GetStateValidity.Request()
        req.group_name = self.group_name
        req.robot_state = moveit_msgs.msg.RobotState()
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = list(q)
        req.robot_state.joint_state.velocity = list(numpy.zeros(self.num_joints))
        req.robot_state.joint_state.effort = list(numpy.zeros(self.num_joints))
        req.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()

        res = self.state_valid_service.call(req)

        return res.valid

    def is_collision_free(self, q_close, q_target):
        bool=True
        q_s=0.01
        M=[]
        v=to_vector(q_target,q_close)
        u=abs(v)
        s=numpy.true_divide(u,q_s)
        t=numpy.ceil(s)
        n_points=max(t)
        #self.get_logger().info('n_points= '+str(n_points))
        step=numpy.true_divide(v,n_points)
        for i in range(int(n_points)):
            M.append((step*i+q_close))

        for p in M:
            if self.is_state_valid(p)==False:
                bool=False
            break
            
        return bool
    
    def to_trajectory(self,path):
        Traj=JointTrajectory()
        for i in range(len(path)):
            p=JointTrajectoryPoint()
            p.positions=list(path[i])
            p.velocities=[]
            p.accelerations=[]
            Traj.points.append(p)
        Traj.joint_names=self.joint_names
        return Traj




class RRTBranch(object):
    '''This is a class which you can use to keep track of your tree branches.
    It is easiest to do this by appending instances of this class to a list 
    (your 'tree'). The class has a parent field and a joint position field (q). 
    
    You can initialize a new branch like this:
        RRTBranch(parent, q)
    Feel free to keep track of your branches in whatever way you want - this
    is just one of many options available to you.
    '''
    def __init__(self, parent, q):
        self.parent = parent
        self.q = q


def main(args=None):
    rclpy.init(args=args)
    ma = MoveArm()
    ma.get_logger().info("Move arm initialization done")
    executor = MultiThreadedExecutor()
    executor.add_node(ma)
    executor.spin()
    ma.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        

