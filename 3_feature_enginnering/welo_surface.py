import numpy as np


class WEloRatingSystem:
    def __init__(self, k=32):
        self.hard_ratings = {}
        self.grass_ratings = {}
        self.clay_ratings = {}
        self.k = k
        self.match_counts = {}

    def check_player(self, player_id, rating=1500, m=0):
        if player_id not in self.hard_ratings:
            self.hard_ratings[player_id] = rating
            self.grass_ratings[player_id] = rating
            self.clay_ratings[player_id] = rating
            self.match_counts[player_id] = m

    def expected_score(self, player_rating, opponent_rating):
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))


    def predict_outcome(self, player, opponent, surface_hard, surface_grass, surface_clay):
        self.check_player(player)
        self.check_player(opponent)

        if surface_hard:
            player_rating = self.hard_ratings[player]
            opponent_rating = self.hard_ratings[opponent]
            expected = self.expected_score(player_rating, opponent_rating)
        elif surface_grass:
            player_rating = self.grass_ratings[player]
            opponent_rating = self.grass_ratings[opponent]
            expected = self.expected_score(player_rating, opponent_rating)
        elif surface_clay:
            player_rating = self.clay_ratings[player]
            opponent_rating = self.clay_ratings[opponent]
            expected = self.expected_score(player_rating, opponent_rating)
        return (player_rating,opponent_rating,expected)

    def update_rating(self, winner_id, loser_id, outcome, winner_game, loser_game,surface_hard, surface_grass, surface_clay ):
        k_winner = self.dynamic_k(self.match_counts[winner_id])
        k_loser = self.dynamic_k(self.match_counts[loser_id])
        f_G = self.match_feature(winner_game, loser_game)
        player_rating_b, opponent_rating_b, expected = self.predict_outcome(winner_id, loser_id, surface_hard, surface_grass, surface_clay)
        if surface_hard:
            self.hard_ratings[winner_id] = player_rating_b + k_winner * f_G * (outcome - expected)
            self.hard_ratings[loser_id] = opponent_rating_b + k_loser * f_G * (outcome - expected)
            self.grass_ratings[winner_id] = self.hard_ratings[winner_id] * 0.61
            self.grass_ratings[loser_id] = self.hard_ratings[loser_id] * 0.61
            self.clay_ratings[winner_id] = self.hard_ratings[winner_id] * 0.45
            self.clay_ratings[loser_id] = self.hard_ratings[loser_id] * 0.45
        elif surface_grass:
            self.grass_ratings[winner_id] = player_rating_b + k_winner * f_G * (outcome - expected)
            self.grass_ratings[loser_id] = opponent_rating_b + k_loser * f_G * (outcome - expected)
            self.hard_ratings[winner_id] = self.grass_ratings[winner_id] * 0.61
            self.hard_ratings[loser_id] = self.grass_ratings[loser_id] * 0.61
            self.clay_ratings[winner_id] = self.grass_ratings[winner_id] * 0.21
            self.clay_ratings[loser_id] = self.grass_ratings[loser_id] * 0.21
        elif surface_clay:
            self.clay_ratings[winner_id] = player_rating_b + k_winner * f_G * (outcome - expected)
            self.clay_ratings[loser_id] = opponent_rating_b + k_loser * f_G * (outcome - expected)
            self.hard_ratings[winner_id] = self.clay_ratings[winner_id] * 0.45
            self.hard_ratings[loser_id] = self.clay_ratings[loser_id] * 0.45
            self.grass_ratings[winner_id] = self.clay_ratings[winner_id] * 0.21
            self.grass_ratings[loser_id] = self.clay_ratings[loser_id] * 0.21

        self.match_counts[winner_id] += 1
        self.match_counts[loser_id] += 1

    def dynamic_k(self, player):
        m = self.match_counts
        return self.k / (1 + np.log(1 + m))

    def match_feature(self, wg, lg):
        return wg / (wg + lg)

