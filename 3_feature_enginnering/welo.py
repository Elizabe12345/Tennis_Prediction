import numpy as np


class WEloRatingSystem:
    def __init__(self, k=32):
        self.ratings = {}
        self.k = k
        self.match_counts = {}

    def add_player(self, player_id, rating=1500, m=0):
        if player_id not in self.ratings:
            self.ratings[player_id] = rating
            self.match_counts[player_id] = m

    def expected_score(self, player_rating, opponent_rating):
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))


    def predict_outcome(self, player, opponent):
        if player not in self.ratings:
            self.add_player(player)
        if opponent not in self.ratings:
            self.add_player(opponent)

        player_rating = self.ratings[player]
        opponent_rating = self.ratings[opponent]
        expected = self.expected_score(player_rating, opponent_rating)
        return expected

    def update_rating(self, winner_id, loser_id, outcome, winner_game, loser_game, ):
        player_rating = self.ratings[winner_id]
        opponent_rating = self.ratings[loser_id]
        k_winner = self.dynamic_k(self.match_counts[winner_id])
        k_loser = self.dynamic_k(self.match_counts[loser_id])
        expected = self.expected_score(player_rating, opponent_rating)
        f_G = self.match_feature(winner_game, loser_game)
        self.ratings[winner_id] = player_rating + k_winner * f_G * (outcome - expected)
        self.ratings[loser_id] = opponent_rating + k_loser * f_G * (outcome - expected)
        self.match_counts[winner_id] += 1
        self.match_counts[loser_id] += 1

    def dynamic_k(self, player):
        m = self.match_counts.get(player, 0)  # 防止 KeyError
        return self.k / (1 + np.log(1 + m))

    def match_feature(self, wg, lg):
        return wg / (wg + lg)

