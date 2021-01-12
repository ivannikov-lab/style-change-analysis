"""
from src.algorithms.threshold_clustering.config import MAX_DISTANCE
average_distance_drop_threshold = 20


class Cluster:
    members = []
    average_distance = 0#MAX_DISTANCE

    def __init__(self, node, distance): # at first cluster distance should be zero or maximum?
        self.members = []
        self.average_distance = 0
        sum = self.average_distance * len(self.members) + distance
        self.average_distance = sum/(len(self.members)+1)
        self.members.append(node)

    def init_two(self, node1,node2, distance):
        self.__init__(node1,distance)
        self.add_member(node2,0)

    def member_is_present(self, node):
        return node in self.members

    def add_member(self, node, distance): # sum of distances from all neighbours
        print("Cluster", self.members, " having avg dist: ",self.average_distance," adding new member ", node)
        if self.member_is_present(node):
            return
        if len(self.members)>=1 and not self.decrease_in_average_acceptable(distance, average_distance_drop_threshold):
            print("cluster average distance drop threshold exceeded")
            return
        sum = self.average_distance * len(self.members) + distance
        self.average_distance = sum / (len(self.members) + 1)
        self.members.append(node)
        # print("total cluster dist: ",sum, len(self.members), " avg:", self.average_distance)
        #print( "updated avg:", self.average_distance)

    def decrease_in_average_acceptable(self, distance, percentage):
        sum_prev = self.average_distance * len(self.members)
        sum_new = sum_prev + distance
        avg_prev = self.average_distance
        avg_new = sum_new/(len(self.members)+1)
        diff = avg_new - avg_prev
        if ((diff*100)/avg_prev <= percentage):
            return True
        else:
            return False

    def merge_with_cluster(self, anotherCluster):         # delete at specific location at another cluster
        sum = self.average_distance * len(self.members) + anotherCluster.average_distance * len(anotherCluster.members)
        self.members.append(anotherCluster.members)
        self.average_distance = self.average_distance/len(self.members)


        """

