
import os,sys,random
from psychopy import prefs
prefs.hardware['audioLib'] =['PTB', 'sounddevice', 'pyo', 'pygame']
from psychopy import visual,core,event,monitors,sound,info
from pandas import read_csv
from psychopy.iohub.client import launchHubServer

import numpy as np; import pandas as pd
from random import shuffle

io=launchHubServer()

testing = int(input("Testing? "))

if testing is not 1 and testing is not 0:
    sys.exit('Enter 0 or 1.')

exp_param_directory = ('/home/coaxlab/Dropbox/metaloki/experimental_parameters/')
graph_array_dir = (os.path.join(exp_param_directory, 'deterministic_graphs/graph_arrays/'))
test_dir = (os.path.join(exp_param_directory, 'testing/'))

data_directory = ('/home/coaxlab/Dropbox/metaloki/data/')

n_phase_trials = 402

if testing:
    subj_id = 'test'

    exp_param_file = (test_dir + 'baseline_graph.npy')
    rot_exp_param_file = (test_dir + 'baseline_graph_rotated.npy')
    unrelated_exp_param_file = (graph_array_dir + 'unrelated_graph.npy')

    n_baseline_trials = 6
    n_rotation_trials = 6
    n_unrelated_trials = 6

else:
    subj_id = input("CoAx ID: ")

    exp_param_file = (graph_array_dir + 'baseline_graph.npy')
    rot_exp_param_file = (graph_array_dir + 'baseline_graph_rotated.npy')
    unrelated_exp_param_file = (graph_array_dir + 'unrelated_graph.npy')

    n_baseline_trials = n_phase_trials # revised n for baseline testing
    n_rotation_trials = n_phase_trials
    n_unrelated_trials = n_phase_trials

output_file_name = 'sub-{}_graph_beh.csv'.format(subj_id)
data_path = data_directory + output_file_name
run_info_path = data_directory + 'sub-{}_graph_runInfo.csv'.format(subj_id)

if not testing and os.path.exists(data_path):
    sys.exit(output_file_name + " already exists!")


# specify constants
exp_param = np.load(exp_param_file) # reward graph
int_reward_graph = exp_param.astype(int)
rot_exp_param = np.load(rot_exp_param_file)
unrelated_exp_param = np.load(unrelated_exp_param_file)

int_rot_reward_graph = rot_exp_param.astype(int)
# n_trials = 180 # planned n for baseline testing in full exp.
block_length = 6
n_total_trials = int(n_baseline_trials + n_rotation_trials + n_unrelated_trials)
# n_blocks = n_trials // block_length
n_blocks = n_total_trials // block_length
start_key = -2
start_position = (np.where(int_reward_graph == start_key)[0][0],np.where(int_reward_graph == start_key)[1][0])

total_reward = 0
int_reward_graph[start_position[0], start_position[1]] = 0
int_rot_reward_graph[start_position[0], start_position[1]] = 0

fast_trial_code, slow_trial_code = -1, -2

instructions_p1 = ("You're going on a treasure hunt! In this hunt, you can choose to open one of four colored doors. Opening one of these doors may reveal a coin you can add to your chest. However, opening the same door will not always give you the same number of coins. \n\nIn fact, there's a thief afoot! The thief sometimes steals the coins you already have. But they don't stop there. Other times, the thief even tries to block your access to what might be behind a given door at a certain point in time. When they block you, the door vanishes. So choose carefully! \n\nPress the spacebar when you're ready to continue.")
instructions_p2 = (" \n\nAfter making your choice, you will receive feedback about how many coins you earned or lost, with a summary of your earnings after every 6 choices. Your goal is to gather as many coins as possible. Press the spacebar to continue.")
instructions_p3 = ("\n\n Choose the yellow door by pressing the yellow button, the blue door with the blue button, the green door with the green button, and the red door with the red button. \n\nNote that if you choose too slowly or too quickly, you won't earn any coins. Finally, remember to make your choice based on the color of the door, and be wary of the thief. \n\nPress the spacebar when you're ready to begin the hunt!")

slow_trial = ("Too slow! \nChoose quickly.")

fast_trial = ("Too fast! \nSlow down. \nYou can continue in 5 seconds.")
break_inst = ("Feel free to take a break! \nPress the green button when you're ready to continue.")


# initialize dependent variables
rt_list = []

coordinate_choice_list = []; coordinate_choice_list.append(start_position) # for first trial navigation
action_choice_list = []

accuracy_list = []

# instantiate psychopy object instances
clock = core.Clock()
expTime_clock = core.Clock()
trialTime_clock = core.Clock()

testing_monitor = monitors.Monitor('testing_computer')
testing_monitor.setSizePix = [1920,1080]
testing_monitor.saveMon()


screen_size = testing_monitor.setSizePix
center=[0,0]

if screen_size != testing_monitor.setSizePix:
    center[0] = (testing_monitor.setSizePix[0]/2) - (screen_size[0]/2)
    center[1] = (testing_monitor.setSizePix[1]/2) - (screen_size[1]/2)

window = visual.Window(size = screen_size, units='pix',monitor = testing_monitor, color = [-1,-1,-1], fullscr=True, pos=center)
inst_msg = visual.TextStim(win=window, units='pix',antialias='False', wrapWidth=screen_size[0]-400, height=screen_size[1]/25)
end_msg = visual.TextStim(win=window, units='pix', antialias='False', wrapWidth=screen_size[0]-400, height=screen_size[1]/25)
break_msg = visual.TextStim(win=window, units='pix', antialias='False', wrapWidth=screen_size[0]-400, height=screen_size[1]/25, pos=[center[0], center[1] + 150])
speed_msg = visual.TextStim(win=window, units='pix',antialias='False', text=slow_trial,  wrapWidth=screen_size[0]-400, height=screen_size[1]/15,
alignHoriz='center', colorSpace='rgb',color=[1,-1,-1], bold=True)


yellow_door = visual.ImageStim(window, image='./images/yellow_door.jpg', units='pix', size=(200,600))
blue_door = visual.ImageStim(window, image='./images/blue_door.jpg', units='pix',  size=(200,600))
green_door = visual.ImageStim(window, image='./images/green_door.jpg', units='pix',  size=(200,600))
red_door = visual.ImageStim(window, image='./images/red_door.jpg', units='pix',  size=(200,600))



choice_emphasis = visual.Rect(win=window, units='pix', height = 650, width= 250, lineColorSpace='rgb',lineColor=[1,1,1], lineWidth=3)
coin = visual.ImageStim(window, image='./images/coin.png',units='pix',size=[screen_size[0]/25], pos=[0,100])
treasure_chest = visual.ImageStim(window, image='./images/treasure_chest.png',units='pix',size=[screen_size[0]/18], pos=[800,screen_size[1]/3.2])

cost_per_decision = 0

runtimeInfo = info.RunTimeInfo(author='kb',win=window,userProcsDetailed=False, verbose=True)
rewardMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[-20,200], colorSpace='rgb', color=[1,1,1],height=screen_size[0]/30)
totalMsg = visual.TextStim(win=window,units='pix',antialias='False', pos=[800,screen_size[1]/2.5],
 colorSpace='rgb', color=[1,1,1],height=screen_size[0]/40)

cue_list = [red_door, green_door, blue_door, yellow_door]


# define target coordinates
red_pos_x = center[0]-500
green_pos_x = center[0]-200
blue_pos_x = center[0] + 100
yellow_pos_x = center[0]+ 400


y = 0

yellow_pos = [yellow_pos_x,y]
blue_pos = [blue_pos_x,y]
green_pos = [green_pos_x,y]
red_pos = [red_pos_x,y]


# set timing variables
fb_time = .9

iti_min = .25
iti_max = .75
fast_penalty_time = 5

n_slow_trials = 1

rt_max = 1
rt_min = .1

yellow_key = 'f'
blue_key = 'd'
green_key = 's'
red_key = 'a'


escape_key = "escape"
inst_key= "space"

# initalize lists
iti_list = []
received_rewards = []
total_rewards = []
trial_time = []
trial_list = []
button_choice_list = []
block_n_list = []
cum_reward_list = []

#give instructions
instruction_phase = True
while instruction_phase:
    inst_msg.text = instructions_p1
    inst_msg.setAutoDraw(True)
    window.flip()
    inst_keys = event.waitKeys(keyList=[inst_key],clearEvents=True)
    inst_msg.text = instructions_p2
    inst_msg.setAutoDraw(True)
    window.flip()
    inst_keys = event.waitKeys(keyList=[inst_key],clearEvents=True)
    inst_msg.text = instructions_p3
    inst_msg.setAutoDraw(True)
    window.flip()
    inst_keys = event.waitKeys(keyList=[inst_key],clearEvents=True)
    instruction_phase = False

inst_msg.setAutoDraw(False)

window.flip()




t = 0
block_n = 0
cum_reward = 0
totalMsg.text = str(total_reward)
totalMsg.setAutoDraw(True)

treasure_chest.setAutoDraw(True)


expTime_clock.reset() # reset so that inst. time is not included
trialTime_clock.reset()
event.clearEvents()


# present choices
while t < n_total_trials:

    if t >= n_baseline_trials:
        int_reward_graph = rot_exp_param.astype(int)
    if t >= (n_baseline_trials + n_rotation_trials):
        int_reward_graph = unrelated_exp_param.astype(int)

    block_n_list.append(block_n)

    if((t%block_length == 0) & (t != 0)):
        block_n += 1
        break_msg.text = ("You earned " + str(int(total_reward)) + " coins this round. \n\nNow let's go back to the beginning and give it another go! Don't let that thief get your hard-earned coin. \n\nPress the spacebar to continue.")
        break_msg.setAutoDraw(True)
        treasure_chest.setPos([0,-200])
        treasure_chest.size = (screen_size[0]/8)
        treasure_chest.setAutoDraw(True)
        coin.setPos([40, -50])
        coin.size = [screen_size[0]/35]
        coin.setAutoDraw(True)
        rewardMsg.setAutoDraw(False)
        totalMsg.setPos([-20,-50])
        totalMsg.text = str(cum_reward_list[-1])
        totalMsg.setAutoDraw(True)
        window.flip()
        break_data = event.waitKeys(keyList=[inst_key], timeStamped=clock)
        break_time = break_data[0][1]
        break_msg.setAutoDraw(False)
        totalMsg.setAutoDraw(False)
        totalMsg.setPos([800,screen_size[1]/2.5])
        coin.setAutoDraw(False)
        treasure_chest.pos = [800,screen_size[1]/3.2]
        treasure_chest.size = (screen_size[0]/18)

        if (t >= n_baseline_trials) & (t < (n_baseline_trials + n_rotation_trials)):
            int_reward_graph = rot_exp_param.astype(int)
        elif t >= (n_baseline_trials + n_rotation_trials):
            int_reward_graph = unrelated_exp_param.astype(int)
        else:
            int_reward_graph = exp_param.astype(int)

        int_reward_graph[start_position[0], start_position[1]] = 0
        total_reward = 0 # reset
        coin.size = [screen_size[0]/25]

        # append nan to lists for return to start
        coordinate_choice_list.append(start_position)
        button_choice_list.append(np.nan)
        action_choice_list.append(np.nan)
        received_rewards.append(np.nan)
        total_rewards.append(np.nan)
        rt_list.append(np.nan)
        trial_time.append(np.nan)
        iti_list.append(np.nan)
        trial_list.append(np.nan)
        block_n_list.append(np.nan)

        event.clearEvents()




    yellow_door.setPos(yellow_pos)
    blue_door.setPos(blue_pos)
    green_door.setPos(green_pos)
    red_door.setPos(red_pos)

    cue_list[0].setAutoDraw(True)
    cue_list[1].setAutoDraw(True)
    cue_list[2].setAutoDraw(True)
    cue_list[3].setAutoDraw(True)

    flip_time=window.flip()
    io.clearEvents('all')

    rt = 0
    while rt == 0:
        keys=io.devices.keyboard.waitForKeys(keys=[yellow_key, blue_key, green_key, red_key, escape_key],
          clear=True)
        rt = keys[0].time - flip_time
        choice=keys[0].key


    if choice==escape_key:
        sys.exit('escape key pressed.')

    # figure out which dimension (x,y) shifted according to key press
    if (choice == green_key) or (choice == red_key):
        shift_dimension = 0 # shifting x
    elif (choice == yellow_key) or (choice == blue_key):
        shift_dimension = 1 # shifting y

    # figure out transition value (1,-1)
    if (choice == blue_key) or (choice == red_key):
        transition = -1 # moving up or left
    elif (choice == yellow_key) or (choice == green_key):
        transition = 1 # moving down or right


    last_position = coordinate_choice_list[-1]


    if shift_dimension == 0:
        new_position = [last_position[shift_dimension] + transition, last_position[~shift_dimension]]
    elif shift_dimension == 1:
        new_position = [last_position[~shift_dimension], last_position[shift_dimension] + transition]

    if (new_position[0] >= int_reward_graph.shape[0]) or (new_position[1] >= int_reward_graph.shape[1]) or (new_position[0] < 0) or (new_position[1] < 0): # if outside bounds of graph (wall hit), don't increment
        new_position = last_position

    coordinate_choice_list.append(new_position)
    reward = int_reward_graph[new_position[0], new_position[1]] # find reward in the landscape
    int_reward_graph[new_position[0], new_position[1]] = 0 # remove reward consumed for future trials. now same behavior as wall.

    if choice == red_key:
        chosen_cue = red_door
        action_choice_list.append("U") # up
        button_choice_list.append("R")
        choice_emphasis.setPos(red_pos)
        rewardMsg.setPos([red_pos[0], red_pos[1]+380])
    elif choice == green_key:
        chosen_cue = green_door
        action_choice_list.append("D") # down
        button_choice_list.append("G")
        choice_emphasis.setPos(green_pos)
        rewardMsg.setPos([green_pos[0], green_pos[1]+380])

    elif choice == blue_key:
        chosen_cue = blue_door
        action_choice_list.append("L") # left
        button_choice_list.append("B")
        choice_emphasis.setPos(blue_pos)
        rewardMsg.setPos([blue_pos[0], blue_pos[1]+380])
    elif choice == yellow_key:
        chosen_cue = yellow_door
        action_choice_list.append("R") # right
        button_choice_list.append("Y")
        choice_emphasis.setPos(yellow_pos)
        rewardMsg.setPos([yellow_pos[0], yellow_pos[1]+380])

    # print('reward ', reward, 'last pos: ', last_position, 'shift dim ', shift_dimension, 'action ', action_choice_list[-1], 'button ', button_choice_list[-1])

    if rt < rt_max and rt > rt_min:

        if reward == 0: # hit a wall
            chosen_cue.setAutoDraw(False) # remove cue

        rewardMsg.text = str(reward)
        total_reward += reward
        cum_reward += reward
        cum_reward_list.append(cum_reward)
        if reward == 1:
            coin.setPos([chosen_cue.pos[0], chosen_cue.pos[1] + 380])
            coin.draw()

        received_rewards.append(reward)

        choice_emphasis.draw()
        if reward != 1:
            rewardMsg.draw()
        totalMsg.text = str(total_reward)
        totalMsg.setAutoDraw(True)
        window.flip()
        core.wait(fb_time)

    if rt >= rt_max:

        rewardMsg.setAutoDraw(False)
        window.flip()
        speed_msg.text = slow_trial

        cue_list[0].setAutoDraw(False)
        cue_list[1].setAutoDraw(False)
        cue_list[2].setAutoDraw(False)
        cue_list[3].setAutoDraw(False)
        speed_msg.setAutoDraw(True)
        window.flip()
        core.wait(fb_time)
        speed_msg.setAutoDraw(False)
        received_rewards.append(slow_trial_code)

    # make subjects wait for a relatively long time if they're fast.
    elif rt <= rt_min:

        rewardMsg.setAutoDraw(False)

        window.flip()
        speed_msg.text = fast_trial

        cue_list[0].setAutoDraw(False)
        cue_list[1].setAutoDraw(False)
        cue_list[2].setAutoDraw(False)
        cue_list[3].setAutoDraw(False)
        speed_msg.setAutoDraw(True)
        window.flip()
        core.wait(fast_penalty_time)
        speed_msg.setAutoDraw(False)
        received_rewards.append(fast_trial_code)

    total_rewards.append(total_reward)
    rt_list.append(rt)

    cue_list[0].setAutoDraw(False)
    cue_list[1].setAutoDraw(False)
    cue_list[2].setAutoDraw(False)
    cue_list[3].setAutoDraw(False)
    window.flip()
    clock.reset()

    # #jitter iti
    iti = random.uniform(iti_min, iti_max)
    iti_list.append(iti)
    core.wait(iti)

    trial_time.append(trialTime_clock.getTime())
    trialTime_clock.reset()

    t+=1
    trial_list.append(t)

total_exp_time=expTime_clock.getTime()


# save data
print(len(button_choice_list), len(action_choice_list), len(coordinate_choice_list[1:]), len(received_rewards), len(total_rewards), len(rt_list), len(trial_time), len(iti_list),
len(block_n_list), len(trial_list))

data_di = {'button_choice': button_choice_list,
 'action_choice': action_choice_list,'coordinate_choice': coordinate_choice_list[1:], # indexing bc start point included, making it (t+1,1)
'reward': received_rewards, 'cumulative_reward': total_rewards, 'rt': rt_list, 'total_trial_time': trial_time,
'iti': iti_list, 'block_n': block_n_list, 'trial_n': trial_list}


data_df = pd.DataFrame(data_di)
data_df.to_csv(data_path, index=False)

runtime_data = np.matrix((str(runtimeInfo['psychopyVersion']), str(runtimeInfo['pythonVersion']),
str(runtimeInfo['pythonScipyVersion']),str(runtimeInfo['pythonPygletVersion']),
str(runtimeInfo['pythonPygameVersion']),str(runtimeInfo['pythonNumpyVersion']),str(runtimeInfo['pythonWxVersion']),
str(runtimeInfo['windowRefreshTimeAvg_ms']), str(runtimeInfo['experimentRunTime']),
str(runtimeInfo['experimentScript.directory']),str(runtimeInfo['systemRebooted']),
str(runtimeInfo['systemPlatform']),str(runtimeInfo['systemHaveInternetAccess']), total_exp_time))

runtime_header = ("psychopy_version, python_version, pythonScipyVersion, pyglet_version, pygame_version, numpy_version, wx_version, window_refresh_time_avg_ms,\
begin_time, exp_dir,last_sys_reboot, system_platform, internet_access, total_exp_time")
np.savetxt(run_info_path,runtime_data, header=runtime_header,delimiter=',',comments='',fmt="%s")

totalMsg.text = cum_reward_list[-1]
end_msg.text = ("Awesome! You have " + totalMsg.text + " coins in total. \nThanks for participating. Let the experimenter know that you're finished.")

#dismiss participant
instruction_phase = True
while instruction_phase:
    treasure_chest.setAutoDraw(False)
    totalMsg.setAutoDraw(False)
    end_msg.setAutoDraw(True)
    window.flip()
    end_keys = event.waitKeys(keyList=[escape_key])

    sys.exit('escape key pressed.')
instruction_phase = False
window.flip()
end_msg.setAutoDraw(False)

window.flip()
window.close()
