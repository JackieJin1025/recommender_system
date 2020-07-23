#!/usr/bin/python3
# coding=utf-8


class PersonalRank:
    """PersonalRank算法"""

    def __init__(self):
        self.train_set = None

    def train(self, origin_data):
        """训练

        :param origin_data:list([int,int,int])
            训练集list([user_id,item_id,rating])
        """
        self.train_set = origin_data

    def recommend(self, user):
        pass
