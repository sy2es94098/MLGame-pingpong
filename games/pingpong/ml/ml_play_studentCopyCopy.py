"""
The template of the script for the machine learning process in game pingpong
"""

# Import the necessary modules and classes
from mlgame.communication import ml as comm
import numpy as np
import pickle
from keras.layers import Dense
from keras.models import Sequential

from easy_tf_log import tflog
from datetime import datetime
from keras import callbacks
import os

from karpathy import prepro, discount_rewards

def ml_loop(side: str):
    """
    The main loop for the machine learning process
    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```
    @param side The side which this script is executed for. Either "1P" or "2P".
    """
    H = 200
    D = 8
    resume = False # resume from previous checkpoint?

    RIGHT_ACTION = 2
    LEFT_ACTION = 3

    if resume:
        model = pickle.load(open('save.p', 'rb'))
    else:
        model = Sequential()
        model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

    # initialization of variables used in the main loop
    x_train, y_train, rewards = [],[],[]
    reward_sum = 0
    episode_nb = 0

    # initialize variables
    resume = True
    running_reward = None
    epochs_before_saving = 10
    log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    # load pre-trained model if exist
    if (resume and os.path.isfile('my_model_weights.h5')):
        print("loading previous weights")
        model.load_weights('my_model_weights.h5')
        
    # add a callback tensorboard object to visualize learning
    tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
            write_graph=True, write_images=True)

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False


    def getObs(player):
        observation = []
        observation.append(scene_info['ball'][0])
        observation.append(scene_info['ball'][1])
        observation.append(scene_info['ball_speed'][0])
        observation.append(scene_info['ball_speed'][1])
        observation.append(scene_info['blocker'][0])
        observation.append(scene_info['blocker'][1])
        if player == '1P':
            observation.append(scene_info['platform_1P'][0])
            observation.append(scene_info['platform_1P'][1])
        if player == '2P':
            observation.append(scene_info['platform_2P'][0])
            observation.append(scene_info['platform_2P'][1])
        observation = np.array(observation)
        return observation

    def move_to(player, pred) : #move platform to predicted position to catch ball 
        if player == '1P':
            if scene_info["platform_1P"][0]+20  > (pred-10) and scene_info["platform_1P"][0]+20 < (pred+10): return 0 # NONE
            elif scene_info["platform_1P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left
        else :
            if scene_info["platform_2P"][0]+20  > (pred-10) and scene_info["platform_2P"][0]+20 < (pred+10): return 0 # NONE
            elif scene_info["platform_2P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left

    def ml_loop_for_1P(): 
        if scene_info['status'] == 'GAME_ALIVE':
            reward = 0
        elif scene_info['status'] == 'GAME_1P_WIN':
            reward = 2
        elif scene_info['status'] == 'GAME_DRAW':
            reward = 1
        else:
            reward = -1
        
        #print(reward)

        if scene_info["ball_speed"][1] > 0 : # 球正在向下 # ball goes down
            x = ( scene_info["platform_1P"][1]-scene_info["ball"][1] ) // scene_info["ball_speed"][1] # 幾個frame以後會需要接  # x means how many frames before catch the ball
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x)  # 預測最終位置 # pred means predict ball landing site 
            bound = pred // 200 # Determine if it is beyond the boundary
            if (bound > 0): # pred > 200 # fix landing position
                if (bound%2 == 0) : 
                    pred = pred - bound*200                    
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) : # pred < 0
                if (bound%2 ==1) :
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            return move_to(player = '1P',pred = pred)
        else : # 球正在向上 # ball goes up
            return move_to(player = '1P',pred = 100)



    def ml_loop_for_2P():  # as same as 1P
        if scene_info["ball_speed"][1] > 0 : 
            return move_to(player = '2P',pred = 100)
        else : 
            x = ( scene_info["platform_2P"][1]+30-scene_info["ball"][1] ) // scene_info["ball_speed"][1] 
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x) 
            bound = pred // 200 
            if (bound > 0):
                if (bound%2 == 0):
                    pred = pred - bound*200 
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) :
                if bound%2 ==1:
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            return move_to(player = '2P',pred = pred)

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()
    _score = [0,0]
    _game_over_score = 11
    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()
        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.


        # 3.3 Put the code here to handle the scene information
        
        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            if side == "1P":
                observation = getObs("1P")
                proba = model.predict(np.expand_dims(observation, axis=1).T)
                action = RIGHT_ACTION if np.random.uniform() < proba else LEFT_ACTION
                y = 1 if action == 2 else 0 # 0 and 1 are our labels

                #print('action' + str(action))
                #print('aprob' + str(aprob))
                #print('a' + str(np.random.uniform()))
                
                if action == 2:
                    comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                else:
                    comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})


                # record various intermediates (needed later for backprop)
                x_train.append(observation)
                y_train.append(y)          

                if scene_info['status'] == 'GAME_ALIVE':
                    reward = 0
                    done = False
                elif scene_info['status'] == 'GAME_1P_WIN':
                    reward = 2
                    _score[0] += 1
                elif scene_info['status'] == 'GAME_DRAW':
                    reward = 1
                    _score[0] += 1
                    _score[1] += 1
                else:
                    reward = -1
                    _score[1] += 1
  
                if _score[0] == _game_over_score or _score[1] == _game_over_score:
                    done = True
                else:
                    done = False

                rewards.append(reward)
                reward_sum += reward      

                if done: # an episode finished
                    print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
        
                    # increment episode number
                    episode_nb += 1
                    
                    # training
                    model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, callbacks=[tbCallBack], sample_weight=discount_rewards(rewards, gamma))
                    
                    # Saving the weights used by our model
                    if episode_nb % epochs_before_saving == 0:    
                        model.save_weights('my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
                    
                    # Log the reward
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    tflog('running_reward', running_reward, custom_dir=log_dir)
                    
                    # Reinitialization
                    x_train, y_train, rewards = [],[],[]
                    reward_sum = 0
            else:
                command = ml_loop_for_2P()
                if command == 0:
                    comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
                elif command == 1:
                    comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                else :
                    comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
            


        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue