from env import cheese
from DeepQNetwork import DeepQNetwork
from DeepQNetwork_2 import DeepQNetwork_2
import my_utils
import multiprocessing as mp
import subprocess
import tensorflow as tf





###################################################################
#####################    训练模式     #############################
###################################################################
def run_cheese():
    # 这个用来表示创建文件的文件名
    # 这个需要定期销毁
    n = 0
    #my_utils.removeFileInDir()  #删除txt文件



    ###############局数可以调参
    for episode in range(1000):


        # initial observation

        env.reset()
        #这里是处理好的初始化36

        print(env.action_list)
        #这里是对首先下的第一颗棋子进行处理
        pos=1
        observation=env.trans_2_observation
        action=RL_2.choose_action(observation,env.action_list)
        env.step(action,pos,n,obj)
        #一开始黑方先行（进攻方）

        pos=-1
        count=0
        n+=1
        ####开局的黑子

        while True:

            #print(n)
            observation_1=env.trans_2_observation
            ######################################################################################
            ###################                No.1 第一子                ########################
            ######################################################################################
            # RL choose action based on observation
            # 运用训练好的神经网络的出来预测的值，然后选一个最大的，将它作为现在这个状态的动作选择
            action_1 = RL_1.choose_action(observation_1,env.action_list)
            #输入的是   19*19*17
            #输出的是   361


            #print(action_1)
            # RL take action and get next observation and reward
            # 通过环境获取在当前状态下选取最好动作，下一步的观察结果，奖励和是否中止
            env.step(action_1,pos,n,obj)
            observation_1_=env.trans_2_observation #移动了一步之后的状态
            reward_1=env.reward
            done=env.done
            #这里要对observation_1进行处理，由在文件中读写来实现
            #处理之后observation_1——>>observation_1m   observation_1_——>>observation_1_m
            #处理之后的observation类的shape变成[9,9,4]
            #


            #用于判断下棋方

            count+=1


            #存储当前的状态转移序列
            RL_1.store_transition(observation_1, action_1, reward_1, observation_1_)


            #根据步数判断是否进行学习
            #开始的时候，进行数据存储不进行学习，等到存到一定地步再进行学习
            if (n > 200) and (n % 5 == 0):
                RL_1.learn()
                save_path=RL_1.saver.save(RL_1.sess,"temp8/model1.ckpt")


            n += 1  # 统计删除
            # break while loop when end of this episode

            if done:
                break



            ######################################################################################
            ###################                No.2 第二子                ########################
            ######################################################################################
            observation_2=env.trans_2_observation
            action_2 = RL_2.choose_action(observation_2,env.action_list)
            env.step(action_2, pos, n,obj)
            observation_2_=env.trans_2_observation #移动了一步之后的状态
            reward_2=env.reward
            done=env.done
            count += 1
            RL_2.store_transition(observation_2, action_2, reward_2, observation_2_)
            if (n> 200) and ((n-1)% 5 == 0):
                RL_2.learn()
                save_path = RL_2.saver.save(RL_2.sess, "temp8/model2.ckpt")
            n += 1
            if done:

                break





            ###更换方向
            pos=-pos


    # end of game

    print('game over')






###################################################################
#####################    比赛模式     #############################
###################################################################
def race_mode():
    received_input=input()

    order, filename1, filename2=my_utils.race_fileprocess(received_input)


    observation,_1,_2,_3=my_utils.fileprocess(filename1)

    f1=open(filename2,'w+')


    ##行棋
    if order==1:
        res1=RL_1.choose_action_race(observation)
        for i in range(361):
            f1.write(str(i)+' '+res1[i])

    else:
        res2=RL_2.choose_action_race(observation)
        for i in range(361):
            f1.write(str(i)+' '+res2[i])


    print('done!')






if __name__ == "__main__":



    ###训练模式，输入命令是
    # 1、【强化学习器博弈引擎】DRLINFO：类似NEW命令，是重建模式，创建一局新棋。
    # DRLINFO[PLAYER][OUTFILE][RECORD]
    # PLAYER: Black / White
    # 例：DRLINFO
    # B
    # D:\1.
    # TXT
    # JJIKJL中，T表示训练模式，B表示电脑走黑方；D:\1.
    # TXT代表输出文件，JJIKJL是走棋序列。
    # 2、【博弈引擎强化学习器】博弈引擎收到DRLINFO命令后，用DRLINFO写入文件，通过文本文件向强化学习器返回以下信息：
    #     1919
    # 的16个channel。每个channel表示一种情况的组合：取某个方向（共四个），执黑 / 白，攻 / 守角色。1919
    # 中的数字取值范围为0
    # ~15，表示PX
    # ~DW类型中的一种。（16
    # 行文本）
    #     1919
    # 棋盘。考虑走棋方的一种所有棋子在棋盘上的完整表示。其中，具有走棋权的一方的棋子用“+1”表示；对手的棋子用“-1”表示；无子的位置用“0”表示。注意：当走棋方发生变动时，相当于对整盘棋的输入表示取负。（1
    # 行文本）
    #     候选走法列表。由引擎筛选了Agent合理的走法，过滤掉大多数无意义的空点，进一步缩小Agent的动作空间，进而大大降低学习难度。为了进程间通信方便，将二维空间中的坐标转换成了1维，强化学习器需要进行与编码时一致的解码过程。（1
    # 行文本）
    #     该状态的立即回报。该过程由引擎的DTSS过程完成。其中，考虑攻防2个角度，所以，reward的值为胜（+1），和（0），负（-1）。引擎已经考虑了转换走棋权的时候，会返回正确的胜负关系。（1
    # 行文本）
    # 以上信息共19行，数据采用csv格式（即逗号分隔的文本字符串）保存。



    obj = subprocess.Popen(["v1.exe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
    # maze game

    env = cheese()

    ##下第一个子

    RL_1 = DeepQNetwork(env.n_actions, env.n_features,

                      learning_rate=0.01,

                      reward_decay=0.9,

                      e_greedy=0.9,

                      replace_target_iter=200,

                      memory_size=2000,

                      # output_graph=True

                      )

    ##下第二个子

    RL_2 = DeepQNetwork_2(env.n_actions, env.n_features,

                      learning_rate=0.01,

                      reward_decay=0.9,

                      e_greedy=0.9,

                      replace_target_iter=200,

                      memory_size=2000,

                      # output_graph=True

                      )

    run_cheese()
    # #多进程部分
    # p = mp.Pool(4)
    # for i in range(4):
    #     p.apply_async(run_cheese,args=())
    # print('Waiting for all subprocesses done...')
    # p.close()
    # p.join()
    # print('All subprocesses done.')
    obj.stdin.flush()
    obj.stdin.write('DRLquit\n')
    obj.stdin.close()

    # ###竞赛模式，输入命令是
    # # 1）【博弈引擎强化学习器】DRLVAL：博弈引擎向强化学习器请求（Query）某局面的Q函数输出值。
    # # 例如：DRLVAL B D:\1.TXT JJIKJL。当在博弈引擎中实现MCTS时，需要通过该命令获取DRL学习器的Q网络。
    # # 2）【强化学习器博弈引擎】强化学习器通过将如下内容写入文本文件D:\1.TXT，向博弈引擎输出Q函数值。
    #
    # m_str=input()
    # m_filename = m_str + '.txt'  #这里是相对路径，文件里存着，当前的处理后的局面以及，现在是在下第几个子
    # m_observation,order=my_utils.race_fileprocess(m_filename)
    # #这里还需要一个运用提前训练好的网络
    # #处理一下保存参数的问题



