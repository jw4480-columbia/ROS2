from arm_dynamics_base import ArmDynamicsBase
import numpy as np
from geometry import rot, xaxis, yaxis

class ArmDynamicsStudent(ArmDynamicsBase):

    def get_q(self, state):
        qs=[]
        for i in range(self.num_links):
            qs.append(state[i,0])
        return qs

    def get_dq(self, state):
        dqs=[]
        for i in range(self.num_links):
            dqs.append(state[i+self.num_links,0])
        return dqs
    

    def get_i_Rot_ip(self, state,i):
        qs=self.get_q(state)
        q=qs[i+1]
        R_i=np.array([[np.cos(q),-1*np.sin(q)],[np.sin(q),np.cos(q)]])
        return R_i
    
    def get_i_Rot_w(self, state,i):
        q=self.get_q(state)
        w_R_i=np.eye(2)
        for j in range(i+1):
            R_i=np.array([[np.cos(q[j]),-1*np.sin(q[j])],[np.sin(q[j]),np.cos(q[j])]])
            
            w_R_i=np.dot(w_R_i,R_i)

        i_R_w=np.transpose(w_R_i)

        return i_R_w
    
    def get_w(self, state):
        dqs=self.get_dq(state)
        ws=[ ]
        w=0
        for i in range(self.num_links):
            w=w+dqs[i]
            ws.append(w)

        return ws
     
    def get_Mf1(self,state):
        N=self.num_links
        Mf1=np.zeros((N,2*N))
        for i in range(N):
            Mf1[i,2*i]=1
            if i < N-1:
                R_i=self.get_i_Rot_ip(state,i)
                Mf1[i,2*i+2]=-1*R_i[0,0]
                Mf1[i,2*i+3]=-1*R_i[0,1]
        return Mf1
    
    def get_Mf2(self,state):
        N=self.num_links
        Mf2=np.zeros((N,2*N))
        for i in range(N):
            Mf2[i,2*i+1]=1
            if i < N-1:
                R_i=self.get_i_Rot_ip(state,i)
                Mf2[i,2*i+2]=-1*R_i[1,0]
                Mf2[i,2*i+3]=-1*R_i[1,1]
        return Mf2
    
    def get_Mf3(self,state):
        N=self.num_links
        Mf3=np.zeros((N,2*N))
        ls=self.link_lengths
        for i in range(N):
            l_i=ls[i]
            Mf3[i,2*i+1]=-0.5*l_i
            if i < N-1:
                R_i=self.get_i_Rot_ip(state,i)
                Mf3[i,2*i+2]=-0.5*l_i*R_i[1,0]
                Mf3[i,2*i+3]=-0.5*l_i*R_i[1,1]
        return Mf3
    
    def get_Mf4_5(self):
        N=self.num_links
        Mf4=np.zeros((N,2*N))
        Mf5=np.zeros((N,2*N))
        return Mf4,Mf5 
    
    def get_Ma1(self):
        N=self.num_links
        ms=self.link_masses
        Ma1=np.zeros((N,2*N))
        for i in range(N):
            Ma1[i,2*i]=-1*ms[i]
            
        return Ma1
    
    def get_Ma2(self):
        N=self.num_links
        ms=self.link_masses
        Ma2=np.zeros((N,2*N))
        for i in range(N):
            Ma2[i,2*i+1]=-1*ms[i]
            
        return Ma2
    
    def get_Ma3(self):
        N=self.num_links
        Ma3=np.zeros((N,2*N))
            
        return Ma3
    
    def get_Ma4(self,state):
        N=self.num_links
        ls=self.link_lengths
        Ma4=np.zeros((N,2*N))
        for i in range(N):
            Ma4[i,2*i]=-1
            if i>0:
                l_mi=ls[i-1]
                mi_R_i=self.get_i_Rot_ip(state,i-1)
                i_R_mi=np.transpose(mi_R_i)
                Ma4[i,2*i-2]=l_mi*i_R_mi[0,0]
                Ma4[i,2*i-1]=l_mi*i_R_mi[0,1]
            
        return Ma4
    
    def get_Ma5(self,state):
        N=self.num_links
        ls=self.link_lengths
        Ma5=np.zeros((N,2*N))
        for i in range(N):
            Ma5[i,2*i+1]=-1
            if i>0:
                l_mi=ls[i-1]
                mi_R_i=self.get_i_Rot_ip(state,i-1)
                i_R_mi=np.transpose(mi_R_i)
                Ma5[i,2*i-2]=l_mi*i_R_mi[1,0]
                Ma5[i,2*i-1]=l_mi*i_R_mi[1,1]
            
        return Ma5
    
    def get_Mdw1(self):
        N=self.num_links
        Mdw1=np.zeros((N,N))
        return Mdw1
    
    def get_Mdw2(self):
        N=self.num_links
        ms=self.link_masses
        ls=self.link_lengths
        Mdw2=np.zeros((N,N))
        for i in range(N):
            m_i=ms[i]
            l_i=ls[i]
            Mdw2[i,i]=-0.5*m_i*l_i
        return Mdw2
    
    def get_Mdw3(self):
        N=self.num_links
        ms=self.link_masses
        ls=self.link_lengths
        Mdw3=np.zeros((N,N))
        for i in range(N):
            m_i=ms[i]
            l_i=ls[i]
            Mdw3[i,i]=-m_i*l_i*l_i/12
        return Mdw3
    
    def get_Mdw4(self,state):
        N=self.num_links
        ls=self.link_lengths
        Mdw4=np.zeros((N,N))
        for i in range(N-1):
            l_i=ls[i]
            i_R_pi=self.get_i_Rot_ip(state,i)
            pi_R_i=np.transpose(i_R_pi)
            Mdw4[i+1,i]=l_i*pi_R_i[0,1]
            
        return Mdw4
    
    def get_Mdw5(self,state):
        N=self.num_links
        ls=self.link_lengths
        Mdw5=np.zeros((N,N))
        for i in range(N-1):
            l_i=ls[i]
            i_R_pi=self.get_i_Rot_ip(state,i)
            pi_R_i=np.transpose(i_R_pi)
            Mdw5[i+1,i]=l_i*pi_R_i[1,1]
            
        return Mdw5

    def get_MC1(self,state):
        N=self.num_links
        MC1=np.zeros((N,1))
        ms=self.link_masses
        ls=self.link_lengths
        g=9.81
        ws=self.get_w(state)
        for i in range(N):
            m_i=ms[i]
            l_i=ls[i]
            w_i=ws[i]
            i_R_w=self.get_i_Rot_w(state,i)
            MC1[i,0]=-0.5*m_i*l_i*w_i*w_i+m_i*g*i_R_w[0,1]
        
        return MC1
    
    def get_MC2(self,state):
        N=self.num_links
        MC2=np.zeros((N,1))
        ms=self.link_masses
        g=9.81
        for i in range(N):
            m_i=ms[i]
            i_R_w=self.get_i_Rot_w(state,i)
            MC2[i,0]=m_i*g*i_R_w[1,1]
        
        return MC2
    
    def get_MC3(self,state,action):
        N=self.num_links
        MC3=np.zeros((N,1))
        dqs=self.get_dq(state)
        mus=action
        mu_pi=0
        psi=self.joint_viscous_friction
        for i in range(N):
            dq_i=dqs[i]
            mu_i=mus[i]
            if i<N-1:
                mu_pi=mus[i+1]
            MC3[i,0]=-mu_i+mu_pi+psi*dq_i
        
        return MC3
    
    def get_MC4(self,state):
        N=self.num_links
        MC4=np.zeros((N,1))
        ls=self.link_lengths
        ws=self.get_w(state)
        for i in range(N-1):
            l_i=ls[i]
            w_i=ws[i]
            i_R_pi=self.get_i_Rot_ip(state,i)
            pi_R_i=np.transpose(i_R_pi)
            MC4[i+1,0]=l_i*pi_R_i[0,0]*w_i*w_i
        
        return MC4
    
    def get_MC5(self,state):
        N=self.num_links
        MC5=np.zeros((N,1))
        ls=self.link_lengths
        ws=self.get_w(state)
        for i in range(N-1):
            l_i=ls[i]
            w_i=ws[i]
            i_R_pi=self.get_i_Rot_ip(state,i)
            pi_R_i=np.transpose(i_R_pi)
            MC5[i+1,0]=l_i*pi_R_i[0,1]*w_i*w_i
        
        return MC5
    
    def get_Mat_n(self,Mf,Ma,Mdw):
        Mn=np.concatenate((Mf,Ma),axis=1)
        Mn=np.concatenate((Mn,Mdw),axis=1)
        return Mn
    
    def get_Mat(self,M1,M2,M3,M4,M5):
        M=np.concatenate((M1,M2),axis=0)
        M=np.concatenate((M,M3),axis=0)
        M=np.concatenate((M,M4),axis=0)
        M=np.concatenate((M,M5),axis=0)
        return M
    


            




    def dynamics_step(self, state, action, dt):
        # state has the following format: [q_0, ..., q_(n-1), qdot_0, ..., qdot_(n-1)] where n is the number of links
        # action has the following format: [mu_0, ..., mu_(n-1)]
        # You can make use of the additional variables:
        # self.num_links: the number of links
        # self.joint_viscous_friction: the coefficient of viscous friction
        # self.link_lengths: an array containing the lengths of all the links
        # self.link_masses: an array containing the masses of all the links

        qs=self.get_q(state)
        dqs=self.get_dq(state)
        N=self.num_links

        
        # Replace this with your code:
        Mf1=self.get_Mf1(state)
        Mf2=self.get_Mf2(state)
        Mf3=self.get_Mf3(state)
        Mf4,Mf5=self.get_Mf4_5()

        Ma1=self.get_Ma1()
        Ma2=self.get_Ma2()
        Ma3=self.get_Ma3()
        Ma4=self.get_Ma4(state)
        Ma5=self.get_Ma5(state)

        Mdw1=self.get_Mdw1()
        Mdw2=self.get_Mdw2()
        Mdw3=self.get_Mdw3()
        Mdw4=self.get_Mdw4(state)
        Mdw5=self.get_Mdw5(state)

        MC1=self.get_MC1(state)
        MC2=self.get_MC2(state)
        MC3=self.get_MC3(state,action)
        MC4=self.get_MC4(state)
        MC5=self.get_MC5(state)

        Mat1=self.get_Mat_n(Mf1,Ma1,Mdw1)
        Mat2=self.get_Mat_n(Mf2,Ma2,Mdw2)
        Mat3=self.get_Mat_n(Mf3,Ma3,Mdw3)
        Mat4=self.get_Mat_n(Mf4,Ma4,Mdw4)
        Mat5=self.get_Mat_n(Mf5,Ma5,Mdw5)

        MRHS=self.get_Mat(MC1,MC2,MC3,MC4,MC5)
        MLHS=self.get_Mat(Mat1,Mat2,Mat3,Mat4,Mat5)

        #print(MRHS)

        x=np.linalg.solve(MLHS,MRHS)
        dws=[]
        for i in range(N):
            dws.append(x[4*N+i,0])
        


        ddqs=[]
        for i in range(N):
            dw_mi=0
            dw_i=dws[i]
            if i>0:
                dw_mi=dws[i-1]
            ddq_i=dw_i-dw_mi
            ddqs.append(ddq_i)

        for i in range(N):
            dq_i_pt=dqs[i]+ddqs[i]*dt
            q_i_pt=qs[i]+dqs[i]*dt+ddqs[i]*dt*dt/2
            state[i]=q_i_pt
            state[N+i]=dq_i_pt
            
        return state

    
