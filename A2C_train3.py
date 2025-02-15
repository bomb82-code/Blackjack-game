import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
#这个程序试图把dealer也弄成A2C程序
from point2 import BlackjackGameTwoPlayers  # 替换为实际游戏文件名
import time

class BlackjackEnv:
    def __init__(self):
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"] * 4  # 模拟牌堆，简化为数字代表牌面大小，10代表花脸牌
        random.shuffle(self.deck)
        self.player_hand = []
        self.dealer_hand = []
        self.is_done = False

    def deal_initial_cards(self):
        self.player_hand.append(self.deck.pop())
        self.dealer_hand.append(self.deck.pop())
        self.player_hand.append(self.deck.pop())
        self.dealer_hand.append(self.deck.pop())

    def step(self, player_action,dealer_action):##就是我的程序中的player_action，get_state和get_game_result的合体。

        if player_action ==1 and dealer_action ==1:# 停牌
            self.is_done = True
            player_value = self.get_hand_value(self.player_hand)
            dealer_value = self.get_hand_value(self.dealer_hand)
            if (dealer_value > 21 and player_value > 21) or player_value == dealer_value:
                return self.get_player_state(), self.get_dealer_state(), 0, 0  #
            elif dealer_value > 21 or (player_value < 21 and player_value > dealer_value):
                return self.get_player_state(),self.get_dealer_state(), 1, -1  # 玩家赢，奖励为1
            elif player_value > 21 or (dealer_value < 21 and player_value < dealer_value):
                return self.get_player_state(), self.get_dealer_state(), -1, 1  # 玩家输，奖励为 -1
            else:
                return self.get_player_state(), self.get_dealer_state(), 0, 0  # 平局，奖励为0


        else:#要牌
            if player_action == 0:  # 要牌
                self.player_hand.append(self.deck.pop())
                #if self.get_hand_value(self.player_hand) > 21:
                    #self.is_done = True
                    #return self.get_player_state(),self.get_dealer_state(), -1, 1 # 爆牌，奖励为 -1
                #return self.get_state(), 0,0
            if dealer_action == 0:  # 要牌
                self.dealer_hand.append(self.deck.pop())
                #if self.get_hand_value(self.dealer_hand) > 21:
                #   self.is_done = True
                #    return self.get_player_state(),self.get_dealer_state(), 1, -1 # 爆牌，奖励为 -1
            #self.is_done = False
            return self.get_player_state(),self.get_dealer_state(), 0, 0



            #while self.get_hand_value(self.dealer_hand) < 17:
             #   self.dealer_hand.append(self.deck.pop())



    def get_player_state(self):
        return (self.get_hand_value(self.player_hand), self.dealer_hand[0])

    def get_dealer_state(self):
        return (self.get_hand_value(self.dealer_hand), self.player_hand[0])

    def get_hand_value(self, hand):
        value = 0

        for card in hand:  # 遍历每张手牌
            if card in ["J", "Q", "K"]:  # J、Q、K 的点数为10
                value += 10
            elif card == "A":  # A 默认值为11
                value += 11
            else:  # 数字牌的点数为其本身
                value += card

        num_aces = hand.count('A') #数一下有多少个A，A可能是21或者是1.
        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1
        return value

    def reset(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        random.shuffle(self.deck)
        self.player_hand = []
        self.dealer_hand = []
        self.is_done = False
        self.deal_initial_cards()

        return self.get_player_state(), self.get_dealer_state()

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

game = BlackjackGameTwoPlayers()  # 初始化新一局游戏
class Agent3:##A2C学习
    def __init__(self,actor,critic, player_id,  alpha=0.1, gamma=0.99, epsilon=0.1,learning_rate=0.001 ):
        """
        智能体类
        :param state_size: 状态空间大小
        :param action_size: 动作空间大小
        :param alpha: 学习率
        :param gamma: 折扣因子
        :param epsilon: 探索率
        """
        #self.q_table = qtable
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.action_probs = torch.tensor([[]])
        self.actor = actor
        self.critic = critic
        self.actor_loss = torch.tensor([[]])
        self.critic_loss = torch.tensor([[]])
        self.advantage = 0
        self.value = 0
        self.optimizer_actor = optim.Adam( self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam( self.critic.parameters(), lr=learning_rate)

        self.player_id = player_id
        self.hand = game.get_player_hand(self.player_id)  # 获取玩家手牌和当前得分

    def learn(self, state, action, new_state, reward,done):
        """
        学习并更新 actor 网络。
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        """
        torch.autograd.set_detect_anomaly(True)
        player_sum, dealer_card = state #游戏中的状态数出包括了玩家手牌和庄家明牌。
        #player_sum_new, dealer_card = new_state

        state = torch.FloatTensor(state).unsqueeze(0)

        self.value = self.critic(state)

        #next_state, reward = game.player_action(action.item())
        player_sum_new, dealer_card = new_state  # 游戏中的状态数出包括了玩家手牌和庄家明牌。

        next_state = torch.FloatTensor(new_state).unsqueeze(0)#咱就先用玩家手牌和庄家名牌一起处理一下子。

        next_value = self.critic(next_state)#咱就先看看用玩家总分+对家名牌是不是能胜率高一点。

        self.advantage = reward + self.gamma * next_value * (1 - int(done)) - self.value  #这一手得好好看看。
        self.actor_loss = -torch.log(self.action_probs[0][action]) * self.advantage.detach()
        self.critic_loss = self.advantage.pow(2)
        print('actor_loss',self.actor_loss)
        self.optimizer_actor.zero_grad()
        self.actor_loss.backward()
        self.optimizer_actor.step()


        self.optimizer_critic.zero_grad()
        self.critic_loss.backward()
        self.optimizer_critic.step()

    def get_state(self, game):
        """
        获取当前玩家的状态，包括总点数和庄家名牌。
        """
        hand = game.get_player_hand(self.player_id)  # 获取玩家手牌和当前得分
        score = hand["score"] if hand["score"]<22 else 0 # 如果总点数超过21，则视为0  ？？这句咋回事？
        #print(score)
        dealer_card = hand["upcard"]  # 获取庄家的明牌（假设为第一张牌）
        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card in ["A"]:
            dealer_card = 1
        return score, dealer_card  # 返回状态元组 (玩家总点数, 庄家明牌)
    '''
    def choose_action(self, state):
        """
        选择动作（基于 ε-greedy 策略）
        :param state: 当前状态
        :return: 选择的动作
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.q_table.q_table.shape[1])  # 随机选择
        return self.q_table.get_best_action(state)  # 选择最佳动作
    '''



    def choose_action(self, state): #0是要牌，1是停牌
        """
        根据ε-greedy策略选择动作。
        """
        player_sum, dealer_card = state  # 提取状态信息,总分、庄家明牌
        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card in ["A"]:
            dealer_card = 1


        state = torch.FloatTensor(state).unsqueeze(0)
        self.action_probs = self.actor(state) #求出下一步两种动作的概率。
        action_dist = torch.distributions.Categorical(self.action_probs) #将下一步两种动作的概率归两类。
        action = action_dist.sample() #根据概率取样
        return action.item()

input_size = 2  # 简化的状态表示为玩家手牌值和庄家手牌值
output_size = 2  # 要牌或停牌两种动作

def train_a2c(num_episodes=100, gamma=0.99, learning_rate=0.001):
    player_score = 0
    dealer_score = 0
    start_time = time.time()
    player_actor = Actor(input_size, output_size)
    player_critic = Critic(input_size)
    dealer_actor = Actor(input_size, output_size)
    dealer_critic = Critic(input_size)
    optimizer_player_actor = optim.Adam(player_actor.parameters(), lr=learning_rate)
    optimizer_player_critic = optim.Adam(player_critic.parameters(), lr=learning_rate)
    optimizer_dealer_actor = optim.Adam(dealer_actor.parameters(), lr=learning_rate)
    optimizer_dealer_critic = optim.Adam(dealer_critic.parameters(), lr=learning_rate)
    player1 = Agent3(player_actor,player_critic, 1)
    player2 = Agent3(dealer_actor,dealer_critic, 2)
    for episode in range(num_episodes):
        game = BlackjackGameTwoPlayers()
        new_hand1, new_hand2 = game.dealcard()
        # 获取玩家初始状态
        player_state = player1.get_state(game)  # 玩家1初始状态
        dealer_state = player2.get_state(game)  # 玩家2初始状态
        player_state = torch.FloatTensor(player_state).unsqueeze(0)
        dealer_state = torch.FloatTensor(dealer_state).unsqueeze(0)

        done = False
        playeract = 0
        dealeract = 0
        while not done:

            player_action_probs = player_actor(player_state)
            #print('probs',action_probs)
            player_action_dist = torch.distributions.Categorical(player_action_probs) #这个函数创建类别分布对象，各个类别的概率等于actor（state）输出的两个训练结果值。
            #print("dist",action_dist)
            player_action = player_action_dist.sample() #从类别分布中随机采样。

            dealer_action_probs = dealer_actor(dealer_state)
            # print('probs',action_probs)
            dealer_action_dist = torch.distributions.Categorical(dealer_action_probs)  # 这个函数创建类别分布对象，各个类别的概率等于actor（state）输出的两个训练结果值。
            # print("dist",action_dist)
            dealer_action = dealer_action_dist.sample()  # 从类别分布中随机采样。

            action1 = player_action.item()
            action2 = dealer_action.item()

            new_hand1 = game.player_action(1, 'y' if playeract == 0 else 'n')  # 执行动作,并返回当前手牌和分数。
            new_hand2 = game.player_action(2, 'y' if dealeract == 0 else 'n')  # 执行动作

            if new_hand1["score"] > 21:
                action1 = 1
            if new_hand2["score"] > 21:
                action2 = 1

            #player_next_state,dealer_next_state, player_reward,dealer_reward = env.step(playeract, dealeract)
            # 判断游戏是否结束
            if action1 and action2:  # 如果两名玩家都停牌，游戏结束
                winner = game.get_game_result()  # 获取游戏结果
                if winner["winner"] == 1:
                    player_reward, dealer_reward= 1,-1
                    player_score += 1
                elif winner["winner"] == 2:
                    player_reward, dealer_reward = -1, 1
                    dealer_score += 1
                else:
                    player_reward, dealer_reward = 0, 0
                done = True  # 标记游戏结束
            else:
                player_reward, dealer_reward = 0, 0
            player_next_state = player1.get_state(game)  # 获取新的状态
            dealer_next_state = player2.get_state(game)  # 获取新的状态
            player_next_state = torch.FloatTensor(player_next_state).unsqueeze(0)
            dealer_next_state = torch.FloatTensor(dealer_next_state).unsqueeze(0)

            player_value = player_critic(player_state)
            player_next_value = player_critic(player_next_state)

            player_advantage = player_reward + gamma * player_next_value * (1 - int(done)) - player_value
            player_actor_loss = -torch.log(player_action_probs[0][player_action]) * player_advantage.detach()
            player_critic_loss = player_advantage.pow(2)

            optimizer_player_actor.zero_grad()
            player_actor_loss.backward()
            optimizer_player_actor.step()

            optimizer_player_critic.zero_grad()
            player_critic_loss.backward()
            optimizer_player_critic.step()

            dealer_value = dealer_critic(dealer_state)
            dealer_next_value = dealer_critic(dealer_next_state)

            dealer_advantage = dealer_reward + gamma * dealer_next_value * (1 - int(done)) - dealer_value
            dealer_actor_loss = -torch.log(dealer_action_probs[0][dealer_action]) * dealer_advantage.detach()
            dealer_critic_loss = dealer_advantage.pow(2)

            optimizer_dealer_actor.zero_grad()
            dealer_actor_loss.backward()
            optimizer_dealer_actor.step()

            optimizer_dealer_critic.zero_grad()
            dealer_critic_loss.backward()
            optimizer_dealer_critic.step()

            player_state = player_next_state
            dealer_state = dealer_next_state

        '''
            if episode % 100 == 99:
            print("player_",player_actor_loss)
            print("dealer_", dealer_actor_loss)
            print("player", player_state)
            print("deal", dealer_state)
            print("winner is", player_reward,dealer_reward)
            print(f"已完成第{episode + 1}轮训练")
        '''

    # 保存状态字典
    torch.save(player_actor.state_dict(), 'player_state_dict.pth')
    torch.save(dealer_actor.state_dict(), 'dealer_state_dict.pth')
    end_time = time.time()
    print("A2C程序训练时间为:", end_time - start_time, "秒")
    print(f"完成A2C第{episode + 1}轮训练")
    if player_score > dealer_score:
        torch.save(player_actor.state_dict(), 'model_dict.pth')
        print(f'player save {player_score/(episode + 1)}',f'{dealer_score/(episode + 1)}')
    else:
        torch.save(dealer_actor.state_dict(), 'model_dict.pth')
        print(f'dealer save {dealer_score/(episode + 1)}',f'{player_score/(episode + 1)}')

def test_a2c(env, player_actor,dealer_actor, num_episodes=100): #这个函数用之前豆包生成程序写得，该淘汰了。
    player_score = 0
    dealer_score = 0

    for episode in range(num_episodes):
        player_state,dealer_state = env.reset()

        player_state = torch.FloatTensor(player_state).unsqueeze(0)
        dealer_state = torch.FloatTensor(dealer_state).unsqueeze(0)
        #print(state)
        done = False
        playeract = 0
        dealeract = 0
        while not done:
            if not playeract:
                player_action_probs = player_actor(player_state)
                player_action_dist = torch.distributions.Categorical(player_action_probs) #这个函数创建类别分布对象，各个类别的概率等于actor（state）输出的两个训练结果值。
                player_action = player_action_dist.sample() #从类别分布中随机采样。

            if not dealeract:
                dealer_action_probs = dealer_actor(dealer_state)
                dealer_action_dist = torch.distributions.Categorical(
                    dealer_action_probs)  # 这个函数创建类别分布对象，各个类别的概率等于actor（state）输出的两个训练结果值。
                dealer_action = dealer_action_dist.sample()  # 从类别分布中随机采样。

            player_value, _ = env.get_player_state()
            dealer_value, _ = env.get_dealer_state()

            playeract = player_action.item()
            dealeract = dealer_action.item()

            if player_value > 21:
                playeract = 1
            if dealer_value > 21:
                dealeract = 1

            player_next_state, dealer_next_state, player_reward, dealer_reward = env.step(playeract, dealeract)

            done = env.is_done

            player_next_state = torch.FloatTensor(player_next_state).unsqueeze(0)
            dealer_next_state = torch.FloatTensor(dealer_next_state).unsqueeze(0)

            player_state = player_next_state
            dealer_state = dealer_next_state
        if player_reward == 1:
            player_score += 1
        if dealer_reward == 1:
            dealer_score += 1
        if episode % 100 == 99:
            print("player", player_state)
            print("deal", dealer_state)
            print("winner is", player_reward, dealer_reward)
            print(f"已完成第{episode + 1}轮测试")
    print("player获胜比例", player_score / num_episodes)
    print("dealer获胜比例", dealer_score / num_episodes)

def test_a2c2(player1_actor,player2_actor, num_episodes=1000):
    player1_score = 0
    player2_score = 0
    player1_wins = 0
    player2_wins = 0
    draws = 0  # 统计平局数量

    player1 = Agent3(player1_actor,critic1, 1)
    player2 = Agent3(player2_actor,critic2, 2)
    for episode in range(num_episodes):
        game = BlackjackGameTwoPlayers()
        new_hand1, new_hand2 = game.dealcard()
        # 获取玩家初始状态
        state1 = player1.get_state(game)  # 玩家1初始状态
        state2 = player2.get_state(game)  # 玩家2初始状态
        #player1_state,player2_state = env.reset()

        #print(state)
        done = False
        while not done:
            #print('1牌', new_hand1)
            #print('2牌', new_hand2)
            state1 = torch.FloatTensor(state1).unsqueeze(0)
            state2 = torch.FloatTensor(state2).unsqueeze(0)
            player1_action_probs = player1_actor(state1)
            player1_action_dist = torch.distributions.Categorical(player1_action_probs) #这个函数创建类别分布对象，各个类别的概率等于actor（state）输出的两个训练结果值。
            player1_action = player1_action_dist.sample() #从类别分布中随机采样。
            #print('player1_action',player1_action.item())
            action1 = player1_action.item()

            player2_action_probs = player2_actor(state2)
            player2_action_dist = torch.distributions.Categorical(
                player2_action_probs)  # 这个函数创建类别分布对象，各个类别的概率等于actor（state）输出的两个训练结果值。
            player2_action = player2_action_dist.sample()  # 从类别分布中随机采样。
            #print('player2_action', player2_action.item())
            action2 = player2_action.item()

            #print('action',action1,action2 )
            new_hand1 = game.player_action(1, 'y' if player1_action.item() == 0 else 'n')  # 执行动作,并返回当前手牌和分数。
            new_hand2 = game.player_action(2, 'y' if player2_action.item() == 0 else 'n')  # 执行动作

            if new_hand1["score"] > 21:
                action1 = 1
            if new_hand2["score"] > 21:
                action2 = 1

            # 判断游戏是否结束
            if action1 and action2:  # 如果两名玩家都停牌，游戏结束
                winner = game.get_game_result()  # 获取游戏结果
                if winner["winner"] == 1:
                    player1_wins += 1
                elif winner["winner"] == 2:
                    player2_wins += 1
                else:
                    draws += 1
                done = True  # 标记游戏结束
            else:
                if not action1:  # 判断一下前一次是否是要牌，如果不要牌就不用抽牌了。
                    new_state1 = player1.get_state(game)  # 获取新的状态
                    state1 = new_state1

                if not action2:
                    new_state2 = player2.get_state(game)  # 获取新的状态
                    state2 = new_state2
        if episode % 100 == 99:
            print("player1", new_hand1["hand"])
            print("player2", new_hand2["hand"])
            print("winner is", winner["winner"])
            print(f"已完成第{episode + 1}轮测试")

    print(f"测试2完成 {num_episodes} 局：")
    print(f"玩家1胜利次数：{player1_wins} ({player1_wins / num_episodes * 100:.2f}%)")
    print(f"玩家2胜利次数：{player2_wins} ({player2_wins / num_episodes * 100:.2f}%)")
    print(f"平局次数：{draws} ({draws / num_episodes * 100:.2f}%)")




if __name__ == "__main__":
    env =  BlackjackEnv()

    critic1 = Critic(input_size)#这两行只是占住Agent的位置，实际中用不到的。
    critic2 = Critic(input_size)
    player_actor = Actor(input_size, output_size)
    player_critic = Critic(input_size)
    dealer_actor = Actor(input_size, output_size)
    dealer_critic = Critic(input_size)
    start_time = time.time()
    #train_a2c(player_actor, player_critic,dealer_actor,dealer_critic)
    train_a2c()
    end_time = time.time()
    print("程序运行时间为:", end_time - start_time, "秒")
    player_actor.load_state_dict(torch.load('player_state_dict.pth'))
    dealer_actor.load_state_dict(torch.load('dealer_state_dict.pth'))
    #test
    test_a2c(env, player_actor, dealer_actor) #用豆包原来的代码进行test

    test_a2c2(player_actor, dealer_actor) #用Agent方式进行test



