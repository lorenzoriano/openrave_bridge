import roslib
roslib.load_manifest("openrave_bridge")
import rospy
import openravepy
from openravepy.databases.inversekinematics import InverseKinematicsModel
import tf
from tf import transformations

import pr2_control_utilities
from geometry_msgs.msg import PoseStamped
import numpy as np
from tabletop_object_detector.srv import TabletopDetectionResponse

def transform_relative_pose_for_ik(manip, matrix4, ref_frame, targ_frame):
    robot = manip.GetRobot()

    if ref_frame == "world":
        worldFromRef = np.eye(4)
    else:
        ref = robot.GetLink(ref_frame)
        worldFromRef = ref.GetTransform()

    if targ_frame == "end_effector":
        targFromEE = np.eye(4)
    else:
        targ = robot.GetLink(targ_frame)
        worldFromTarg = targ.GetTransform()
        worldFromEE = manip.GetEndEffectorTransform()
        targFromEE = np.dot(np.linalg.inv(worldFromTarg), worldFromEE)

    refFromTarg_new = matrix4
    worldFromEE_new = np.dot(np.dot(worldFromRef, refFromTarg_new), targFromEE)

    return worldFromEE_new

def tf_for_link(T_w_link, manip, link_name):
    """
    Transforms an arbitrary link attached to the manipulator
    e.g. you might want ik for pr2 "r_gripper_tool_frame" instead of the openrave EE frame
    T_w_link: 4x4 matrix. "world frame from link frame"
    manip: openrave Manipulator
    link_name: (you know)
    filter_options: see openravepy.IkFilterOptions
    """
        
    robot = manip.GetRobot()

    link = robot.GetLink(link_name)

    if not robot.DoesAffect(manip.GetArmJoints()[-1], link.GetIndex()):
        raise Exception("link %s is not attached to end effector of manipulator %s"%(link_name, manip.GetName()))

    Tcur_w_link = link.GetTransform()
    Tcur_w_ee = manip.GetEndEffectorTransform()
    Tf_link_ee = np.linalg.solve(Tcur_w_link, Tcur_w_ee)
    
    T_w_ee = T_w_link.dot(Tf_link_ee)
    return T_w_ee

class PR2Robot(object):
    def __init__(self):
        self.env = openravepy.Environment()
        self.robot = self.env.ReadRobotXMLFile("robots/pr2-beta-sim.robot.xml")
        self.env.Add(self.robot)
        
        rospy.loginfo("Loading IK for rightarm")
        manip = self.robot.SetActiveManipulator("rightarm")
        self.rightarm_ik = InverseKinematicsModel(self.robot, 
                                                  iktype=openravepy.IkParameterization.Type.Transform6D)
        if not self.rightarm_ik.load():
            self.rightarm_ik.autogenerate()
            
        rospy.loginfo("Loading IK for leftarm")
        manip = self.robot.SetActiveManipulator("leftarm")
        self.leftarm_ik = InverseKinematicsModel(self.robot, 
                                                 iktype=openravepy.IkParameterization.Type.Transform6D)
        if not self.leftarm_ik.load():
            self.leftarm_ik.autogenerate()
            
        self.robot_state = pr2_control_utilities.RobotState()
        self.controller = pr2_control_utilities.PR2JointMover(robot_state = self.robot_state,
                                                              name = "PR2 Controller",
                                                              time_to_reach=5.0
                                                              )
        self.listener = tf.TransformListener()
        
        #fixing the joints
        joint_msg = self.robot_state.last_joint_msg
        ros_names = joint_msg.name
        inds_ros2rave = np.array([self.robot.GetJointIndex(name) for name in ros_names])
        self.good_ros_inds = np.flatnonzero(inds_ros2rave != -1) # ros joints inds with matching rave joint
        self.rave_inds = inds_ros2rave[self.good_ros_inds] # openrave indices corresponding to those joints        
        
    
    def convertPose(self, pose):
        assert isinstance(pose, PoseStamped)
        self.listener.waitForTransform("/base_footprint", pose.header.frame_id,
                                       rospy.Time.now(), rospy.Duration(1))
        newpose = self.listener.transformPose("/base_footprint", pose)        
        translation = tf.listener.xyz_to_mat44(newpose.pose.position), 
        orientation = tf.listener.xyzw_to_mat44(newpose.pose.orientation)
        
        matrix4 = np.dot(translation, orientation).squeeze()
        return matrix4
        
    
    def __ik_solution(self, pose, manip, end_effector_link, ignore_end_effector=True):
        T = self.convertPose(pose)
        self.update_rave()
        worldFromEE = tf_for_link(T, manip, end_effector_link)
        filter_options = openravepy.IkFilterOptions.IgnoreEndEffectorCollisions        
        if ignore_end_effector:
            filter_options = filter_options | openravepy.IkFilterOptions.CheckEnvCollisions
        
        sol = manip.FindIKSolution(worldFromEE, filter_options)
        return sol
    
    def find_rightarm_ik(self, pose, ignore_end_effector=True):
        manip = self.robot.SetActiveManipulator("rightarm")
        return self.__ik_solution(pose, manip, "r_wrist_roll_link", ignore_end_effector)
    
    def find_leftarm_ik(self, pose, ignore_end_effector=True):
        manip = self.robot.SetActiveManipulator("leftarm")
        return self.__ik_solution(pose, manip, "l_wrist_roll_link", ignore_end_effector)    
    
    def move_right_arm(self, pose, ignore_end_effector = True):
        sol = self.find_rightarm_ik(pose, ignore_end_effector)
        if sol is None:
            rospy.logerr("Could not find an IK solution!")
            return False
        self.controller.set_arm_state(sol.tolist(), "right", True)
    
    def move_left_arm(self, pose, ignore_end_effector = True):
        sol = self.find_leftarm_ik(pose, ignore_end_effector)
        if sol is None:
            rospy.logerr("Could not find an IK solution!")
            return False
        self.controller.set_arm_state(sol.tolist(), "left", True)
        
    def update_rave(self):
        ros_values = self.robot_state.last_joint_msg.position
        rave_values = [ros_values[i_ros] for i_ros in self.good_ros_inds]
        self.robot.SetJointValues(rave_values[:20],self.rave_inds[:20])
        self.robot.SetJointValues(rave_values[20:],self.rave_inds[20:])

def add_table(table_msg, env, tf_listener):
    assert isinstance(table_msg, TabletopDetectionResponse)
    body = openravepy.RaveCreateKinBody(env,'')
    body.SetName('table')
    z = msg.detection.table.pose.pose.position.z
    x_min = msg.detection.table.x_min
    x_max = msg.detection.table.x_max
    y_min = msg.detection.table.y_min
    y_max = msg.detection.table.y_max
    
    body = openravepy.RaveCreateKinBody(pr2.env,'')
    body.SetName("table")
    x = (x_max-x_min)/2 + x_min
    y = (y_max-y_min)/2 + y_min
    dim_x = (x_max - x_min)/2
    dim_y = (y_max - y_min)/2
    body.InitFromBoxes(np.array([[x, y, z, dim_x, dim_y, 0.01]]), True)
    env.Add(body, True)    
    
    return body