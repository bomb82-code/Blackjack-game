import random  # 导入random模块，用于随机发牌

'''
def deal_card():
    # 定义一副牌，包含数字牌、J、Q、K和A
    cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"]*4

    return random.choice(cards)  # 随机选择并返回一张牌
'''



def calculate_score(hand):
    score = 0  # 初始化总分
    aces = 0  # 记录手牌中A的数量

    for card in hand:  # 遍历每张手牌
        if card in ["J", "Q", "K"]:  # J、Q、K 的点数为10
            score += 10
        elif card == "A":  # A 默认值为11
            score += 11
            aces += 1
        else:  # 数字牌的点数为其本身
            score += card

    # 如果总分大于21且有A，将一个A的点数从11变为1
    while score > 21 and aces > 0:
        score -= 10  # 将A从11改为1
        aces -= 1  # 减少一个A的计数

    return score  # 返回计算后的总分


class BlackjackGameTwoPlayers:
    def __init__(self):
        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"] * 4
        random.shuffle(self.cards)

        # 玩家1和玩家2的初始手牌，各抽两张牌
        self.player1_hand = [random.choice(self.cards)] #这里只是随机初始化一下，避免玩家定义时手牌为空无法使用get_hand函数。
        self.player2_hand = [random.choice(self.cards)]
        # 标记玩家1和玩家2是否结束游戏（停止抽牌或爆牌）
        self.player1_over = False
        self.player2_over = False
    def deal_card(self):
        #print (self.cards)
        return self.cards.pop()


    def dealcard(self):
        self.player1_hand = [self.deal_card(), self.deal_card()]
        self.player2_hand = [self.deal_card(), self.deal_card()]
        return self.get_player_hand(1), self.get_player_hand(2)

    def get_player_hand(self, player_id):   # 随时查看手牌
        hand = self.player1_hand if player_id == 1 else self.player2_hand  # 根据玩家ID获取对应手牌
        upcard=self.player2_hand[0] if player_id == 1 else self.player1_hand[0]
        return {
            "hand": hand,
            "score": self.calculate_score(hand),  # 计算手牌得分
            "upcard": upcard
        }

    def player_action(self, player_id, action):     # 游戏动作接口
        if player_id == 1:
            hand = self.player1_hand  # 获取玩家1的手牌
            self.player1_over = action == "n"  # 如果玩家选择停止，则标记结束
        else:
            hand = self.player2_hand  # 获取玩家2的手牌
            self.player2_over = action == "n"

        if action == "y":  # 如果玩家选择抽牌
            hand.append(self.deal_card())  # 为玩家发一张牌

        return self.get_player_hand(player_id)  # 返回当前玩家状态

    def get_game_result(self):      # 获取结果
        # 计算玩家1和玩家2的最终得分
        player1_score = self.calculate_score(self.player1_hand)
        player2_score = self.calculate_score(self.player2_hand)

        # 构造结果字典，包含双方手牌、分数和胜负信息
        result = {
            "player1_hand": self.player1_hand,
            "player1_score": player1_score,
            "player2_hand": self.player2_hand,
            "player2_score": player2_score,
            "winner":   0   # 3平局
        }

        # 判断胜负逻辑
        if player1_score > 21 and player2_score > 21:
            result["winner"] = 3  # 双方爆牌，平局
        elif player1_score > 21:
            result["winner"] = 2  # 玩家1爆牌，玩家2获胜
        elif player2_score > 21:
            result["winner"] = 1  # 玩家2爆牌，玩家1获胜
        elif player1_score > player2_score:
            result["winner"] = 1  # 玩家1得分更高
        elif player1_score < player2_score:
            result["winner"] = 2  # 玩家2得分更高
        else:
            result["winner"] = 3  # 双方得分相同，平局

        return result  # 返回最终结果

    def calculate_score(self, hand):
        '''
        :param hand:
        :return:

          score = 0  # 初始化总分
            aces = 0  # 记录手牌中A的数量

            for card in hand:  # 遍历每张手牌
                if card in ["J", "Q", "K"]:  # J、Q、K 的点数为10
                    score += 10
                elif card == "A":  # A 默认值为11
                    score += 11
                    aces += 1
                else:  # 数字牌的点数为其本身
                    score += card

            # 如果总分大于21且有A，将一个A的点数从11变为1
            while score > 21 and aces > 0:
                score -= 10  # 将A从11改为1
                aces -= 1  # 减少一个A的计数

            return score  # 返回计算后的总分
        '''
        value = 0
        for card in hand:  # 遍历每张手牌
            if card in ["J", "Q", "K"]:  # J、Q、K 的点数为10
                value += 10
            elif card == "A":  # A 默认值为11
                value += 11
            else:  # 数字牌的点数为其本身
                value += card

        num_aces = hand.count('A')  # 数一下有多少个A，A可能是21或者是1.
        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1
        return value



# 示例外部程序调用
if __name__ == "__main__":
    game = BlackjackGameTwoPlayers()  # 创建游戏实例
    for p_id in [1, 2]:
        print(f"玩家{p_id}回合:")
        if p_id == 1:
            play_over = game.player1_over
        else:
            play_over = game.player2_over
        while not play_over:
            player1_hand_info = game.get_player_hand(1)  # 获取玩家的手牌和分数信息
            player2_hand_info = game.get_player_hand(2)
            if p_id == 1:
                print(f"玩家1手牌:{player1_hand_info['hand']}, 当前得分:{player1_hand_info['score']}")
                print(f"玩家2第一张牌:{player2_hand_info['hand'][0]}")
            else:
                print(f"玩家2手牌:{player2_hand_info['hand']}, 当前得分:{player2_hand_info['score']}")
                print(f"玩家1第一张牌:{player1_hand_info['hand'][0]}")
            action = input(f" 输入 'y' 抽牌 或 'n' 停止抽牌: ").strip().lower()
            response = game.player_action(p_id, action)  # 执行玩家操作
            if p_id == 1:       # 更新over信息
                play_over = game.player1_over
            else:
                play_over = game.player2_over
    re = game.get_game_result()
    print(f"玩家1最终手牌{re['player1_hand']}, 最终得分{re['player1_score']}")
    print(f"玩家2最终手牌{re['player2_hand']}, 最终得分{re['player2_score']}")
    if re['winner'] == 3:
        print("平局！")
    else:
        print(f"玩家{re['winner']}获胜!")
