import numpy as np


class WEloRatingSystem:
    def __init__(self, k=32):
        # 用一个字典管理所有场地的评分，键为场地名称
        self.ratings = {
            "hard": {},
            "grass": {},
            "clay": {}
        }
        self.match_counts = {}
        self.k = k
        # 当某一场地为更新主场地时，其它场地评分根据主场地乘以转换系数更新
        self.conversion = {
            "hard": {"grass": 0.61, "clay": 0.45},
            "grass": {"hard": 0.61, "clay": 0.21},
            "clay": {"hard": 0.45, "grass": 0.21}
        }

    def check_player(self, player_id, rating=1500, m=0):
        """检查玩家是否存在，不存在则初始化所有场地评分和比赛计数"""
        flag = 1
        if player_id not in self.match_counts:
            flag = 0
            for surface in self.ratings:
                self.ratings[surface][player_id] = rating
            self.match_counts[player_id] = m
        return flag

    @staticmethod
    def expected_score(player_rating, opponent_rating):
        """计算 ELO 预期得分"""
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))

    def predict_outcome(self, player, opponent, surface):
        """根据指定场地预测对局结果"""
        self.check_player(player)
        self.check_player(opponent)
        player_rating = self.ratings[surface][player]
        opponent_rating = self.ratings[surface][opponent]
        expected = self.expected_score(player_rating, opponent_rating)
        return player_rating, opponent_rating, expected

    def dynamic_k(self, match_count):
        """根据玩家比赛次数计算动态 K 值"""
        return self.k / (1 + np.log(1 + match_count))

    def match_feature(self, winner_game, loser_game):
        """计算局数特征，避免分母为 0 的情况"""
        total = winner_game + loser_game
        if total == 0:
            return 0.5
        return winner_game / total

    def update_rating(self, winner_id, loser_id, outcome, winner_game, loser_game, surface):
        """
        更新指定场地的 ELO 评分，同时更新其它场地评分。
        参数：
          winner_id, loser_id: 玩家 id
          outcome: 对胜者来说的实际结果（1 表示胜利，0 表示失败）
          winner_game, loser_game: 单场比赛中各自赢得的局数
          surface: 更新时所依据的主场地（"hard", "grass" 或 "clay"）
        """
        # 确保双方存在
        self.check_player(winner_id)
        self.check_player(loser_id)

        # 计算动态 K 值
        k_winner = self.dynamic_k(self.match_counts[winner_id])
        k_loser = self.dynamic_k(self.match_counts[loser_id])
        # 计算局数特征
        f_G = self.match_feature(winner_game, loser_game)

        # 根据主场地预测评分和预期得分
        winner_base, loser_base, expected = self.predict_outcome(winner_id, loser_id, surface)

        # 更新主场地评分
        update = k_winner * f_G * (outcome - expected)
        new_winner_rating = winner_base + update
        new_loser_rating = loser_base + update
        self.ratings[surface][winner_id] = new_winner_rating
        self.ratings[surface][loser_id] = new_loser_rating

        # 根据转换系数更新其它场地评分
        for other_surface, factor in self.conversion[surface].items():
            self.ratings[other_surface][winner_id] += update * factor
            self.ratings[other_surface][loser_id] += update * factor

        # 更新比赛次数
        self.match_counts[winner_id] += 1
        self.match_counts[loser_id] += 1
