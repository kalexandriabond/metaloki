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
key_inversion = int(input("Key inversion? "))
candidate_inversions = [0, 1]

probe_status = int(input("Probing? "))
probe_statuses = [0,1]
# input validation
if testing is not 1 and testing is not 0:
    sys.exit('Enter 0 or 1.')
if key_inversion not in candidate_inversions:
    sys.exit('Enter an inversion of {}'.format(candidate_inversions))
if probe_status not in probe_statuses:
    sys.exit('Enter probe as present or not using {}'.format(probe_statuses))


home = os.path.expanduser('~')
exp_param_directory = os.path.join(home,'Dropbox/metaloki_veiled/experimental_parameters/')
graph_array_dir = (os.path.join(exp_param_directory, 'deterministic_graphs/graph_arrays/'))
test_dir = (os.path.join(exp_param_directory, 'testing/'))


data_directory = os.path.join(home,'Dropbox/metaloki_veiled/data/')


n_phase_trials = 1000 # 100 blocks
n_choices = 10
n_test_rounds = 10

if testing:
    subj_id = 'test'
    n_baseline_trials = n_choices*n_test_rounds

    if probe_status == 1:
        exp_param_file = (test_dir + 'baseline_graph_binary_expanded_long_NO_probe.npy') # changed probe input
    elif probe_status == 0:
        exp_param_file = (test_dir + 'baseline_graph_binary_expanded_long_NO_probe.npy')# changed probe input
    probe_paths_file = os.path.join(test_dir, 'probe_paths.csv')
    probe_timing_file = os.path.join(test_dir, 'probe_timing.csv')
else:
    subj_id = input("CoAx ID: ")

    if probe_status == 1:
        exp_param_file = (graph_array_dir + 'baseline_graph_binary_expanded_long_NO_probe.npy')# changed probe input
    elif probe_status == 0:
        exp_param_file = (graph_array_dir + 'baseline_graph_binary_expanded_long_NO_probe.npy')# changed probe input
    probe_paths_file = os.path.join(graph_array_dir, 'probe_paths.csv')
    probe_timing_file = os.path.join(graph_array_dir, 'probe_timing.csv')

    n_baseline_trials = n_phase_trials # revised n for baseline testing



output_file_name = 'sub-{}_graph_binary_expanded_long_beh_key{}_probe{}.csv'.format(subj_id, key_inversion, probe_status)
data_path = data_directory + output_file_name
run_info_path = data_directory + 'sub-{}_graph_binary_expanded_long_key{}_probe{}_runInfo.csv'.format(subj_id, key_inversion, probe_status)

if not testing and os.path.exists(data_path):
    sys.exit(output_file_name + " already exists!")


# specify constants
exp_param = np.load(exp_param_file) # reward graph
int_reward_graph = exp_param.astype(int)
probe_path_df = pd.read_csv(probe_paths_file)
probe_timing_df = pd.read_csv(probe_timing_file)


block_length = n_choices
n_total_trials = int(n_baseline_trials)
n_blocks = n_total_trials // block_length
start_key = -2
start_position = (np.where(int_reward_graph == start_key)[0][0],np.where(int_reward_graph == start_key)[1][0])

total_reward = 0
int_reward_graph[start_position[0], start_position[1]] = 0

fast_trial_code, slow_trial_code = -1, -2

instructions_p1 = ("You're going on a treasure hunt! In this hunt, you can choose to open one of four colored mystery boxes. Opening one of these boxes may reveal a coin you can add to your treasure chest. However, opening the same mystery box will not always give you the same number of coins. \n\nIn fact, there's a thief afoot! This sinister fellow tries to block your access to what might be within a given box at a certain point in time. When they block you, the box vanishes. So choose carefully! \n\nPress the spacebar when you're ready to continue.")
instructions_p2 = (" \n\nAfter making your choice, you will receive feedback about how many coins you earned or lost, with a summary of your earnings after every 10 choices. Your goal is to gather as many coins as possible. Press the spacebar to continue.")
instructions_p3 = ("\n\n Choose the mystery boxes using the colored buttons in front of you. \n\nNote that if you choose too slowly or too quickly, you won't earn any coins. Finally, remember to make your choice based on the color of the box, and be wary of the thief. \n\nPress the spacebar when you're ready to begin the hunt!")

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


yellow_box = visual.ImageStim(window, image='./images/yellow_box.jpg', units='pix', size=(280,280))
blue_box = visual.ImageStim(window, image='./images/blue_box.jpg', units='pix',  size=(280,280))
green_box = visual.ImageStim(window, image='./images/green_box.jpg', units='pix',  size=(280,280))
red_box = visual.ImageStim(window, image='./images/red_box.jpg', units='pix',  size=(280,280))



choice_emphasis = visual.Rect(win=window, units='pix', height = 320, width= 320, lineColorSpace='rgb',lineColor=[1,1,1], lineWidth=3)
coin = visual.ImageStim(window, image='./images/coin.png',units='pix',size=[screen_size[0]/25], pos=[0,10])
treasure_chest = visual.ImageStim(window, image='./images/treasure_chest.png',units='pix',size=[screen_size[0]/18], pos=[800,screen_size[1]/3.2])

cost_per_decision = 0

runtimeInfo = info.RunTimeInfo(author='kb',win=window,userProcsDetailed=False, verbose=True)
rewardMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[-20,200], colorSpace='rgb', color=[1,1,1],height=screen_size[0]/30)
totalMsg = visual.TextStim(win=window,units='pix',antialias='False', pos=[800,screen_size[1]/2.5],
 colorSpace='rgb', color=[1,1,1],height=screen_size[0]/40)

cue_list = [red_box, green_box, blue_box, yellow_box]


# define target coordinates
red_pos_x = center[0]-600
green_pos_x = center[0]-200
blue_pos_x = center[0] + 200
yellow_pos_x = center[0]+ 600


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


if key_inversion == 0:
    yellow_key = 'f'
    blue_key = 'd'
    green_key = 's'
    red_key = 'a'
elif key_inversion == 1:
    yellow_key = 'a'
    blue_key = 's'
    green_key = 'd'
    red_key = 'f'
else:
    print('Invalid key inversion. Re-enter as 0/1.')

onpath_probe_code, offpath_probe_code, start_code, wall_code, normal_trial_code, probe_free_choice_code = -4, -5, -2, -3, 0, 1

escape_key = "escape"
inst_key = "space"

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
block_trial = 0
cum_reward = 0
reward = 0
totalMsg.text = str(total_reward)
totalMsg.setAutoDraw(True)

treasure_chest.setAutoDraw(True)


expTime_clock.reset() # reset so that inst. time is not included
trialTime_clock.reset()
event.clearEvents()

offpath_probe_count = []
onpath_probe_count = []

last_position = start_position
trial_type_list = []
trial_type = normal_trial_code
# present choices
while t < n_total_trials:
    print('trial ', t)
    last_position = coordinate_choice_list[-1]


    only_direction_avail = 'all'


    if((t%block_length == 0) & (t != 0)):
        block_n += 1
        block_trial = 1

        print('BLOCK NUM ', block_n)

        if (probe_status == 1) & (block_n in probe_timing_df.probe_block_idx.values):
            probe_timing_block_df = probe_timing_df[probe_timing_df.probe_block_idx == block_n].reset_index(drop=True)
            probe_type, probe_id = probe_timing_block_df.probe_type.values[0], probe_timing_block_df.probe_id.values[0]
            probe_block_path_df = probe_path_df[(probe_path_df.probe_type == probe_type) & (probe_path_df.probe_id == probe_id)].reset_index(drop=True)
            block_length = len(probe_block_path_df)
            break_msg.text = ("Now let's go back to the beginning and give it another go! Don't let that thief get your hard-earned coin. \n\nPress the spacebar to continue.")
            break_msg.setAutoDraw(True)

            print('BLOCK LEN ', block_length)
        else:
            block_length = n_choices # reset to 10 choices after a probe block
            break_msg.text = ("You earned " + str(int(total_reward)) + " coins this round. \n\nNow let's go back to the beginning and give it another go! Don't let that thief get your hard-earned coin. \n\nPress the spacebar to continue.")
            break_msg.setAutoDraw(True)
            print("NONPROBE BLOCK")


        # break_msg.text = ("You earned " + str(int(total_reward)) + " coins this round. \n\nNow let's go back to the beginning and give it another go! Don't let that thief get your hard-earned coin. \n\nPress the spacebar to continue.")
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

        int_reward_graph = exp_param.astype(int)

        int_reward_graph[start_position[0], start_position[1]] = 0
        total_reward = 0 # reset
        reward = 0
        coin.size = [screen_size[0]/25]

        last_position = start_position
        # append nan to lists for return to start
        coordinate_choice_list.append(start_position)
        trial_type_list.append(np.nan)
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

    if (probe_status == 1) & (block_n in probe_timing_df.probe_block_idx.values):
        probe_timing_block_df = probe_timing_df[probe_timing_df.probe_block_idx == block_n].reset_index(drop=True)
        probe_type, probe_id = probe_timing_block_df.probe_type.values[0], probe_timing_block_df.probe_id.values[0]
        probe_block_path_df = probe_path_df[(probe_path_df.probe_type == probe_type) & (probe_path_df.probe_id == probe_id)].reset_index(drop=True)
        only_direction_avail = probe_block_path_df.loc[probe_block_path_df.probe_block_trial == block_trial, 'action'].values[0]

        if only_direction_avail == 'all':
            trial_type = probe_free_choice_code
        else:
            trial_type = probe_type # from probe_type
        print('only_direction_avail', only_direction_avail)
        print('block_trial', block_trial)
    else:
        trial_type = normal_trial_code

    print(only_direction_avail, 'only_direction_avail', 'trial_type', trial_type)


    yellow_box.setPos(yellow_pos)
    blue_box.setPos(blue_pos)
    green_box.setPos(green_pos)
    red_box.setPos(red_pos)

    if only_direction_avail == 'U':
        cue_list[0].setAutoDraw(True)
        forced_choice = red_key
        flip_time = window.flip()
    if only_direction_avail == 'D':
        cue_list[1].setAutoDraw(True)
        forced_choice = green_key
        flip_time = window.flip()
    if only_direction_avail == 'L':
        cue_list[2].setAutoDraw(True)
        forced_choice = blue_key
        flip_time = window.flip()
    if only_direction_avail == 'R':
        cue_list[3].setAutoDraw(True)
        flip_time = window.flip()
        forced_choice = yellow_key
    if only_direction_avail == '-1': # they are in the offpath probe quad or diagonal to it (so card. dir. doesn't work)
        print('edge')
        cue_list[0].setAutoDraw(True)
        cue_list[1].setAutoDraw(True)
        cue_list[2].setAutoDraw(True)
        cue_list[3].setAutoDraw(True)
        flip_time = window.flip()
    elif only_direction_avail == 'all':
        print('normal')
        cue_list[0].setAutoDraw(True)
        cue_list[1].setAutoDraw(True)
        cue_list[2].setAutoDraw(True)
        cue_list[3].setAutoDraw(True)
        flip_time = window.flip()
    io.clearEvents('all')

    rt = 0

    if (trial_type == normal_trial_code) or (trial_type == probe_free_choice_code):
        choice_set = [yellow_key, blue_key, green_key, red_key, escape_key]
    elif trial_type != normal_trial_code:
        choice_set = forced_choice

    while rt == 0:
        keys=io.devices.keyboard.waitForKeys(keys=choice_set, clear=True)
        rt = keys[0].time - flip_time
        choice=keys[0].key


    if choice==escape_key:
        sys.exit('escape key pressed.')

    # figure out which dimension (x,y) shifted according to key press
    # remember: navigating an array, not a cart. coordinate system.
    # so shifting dim 0 / x = shifting rows, Up/Down. shifting dim 1 / y = shifting columns, Left/Right.
    if (choice == green_key) or (choice == red_key):
        shift_dimension = 0 # shifting x
    elif (choice == yellow_key) or (choice == blue_key):
        shift_dimension = 1 # shifting y

    # figure out transition value (1,-1)
    if (choice == blue_key) or (choice == red_key):
        transition = -1 # moving up or left
    elif (choice == yellow_key) or (choice == green_key):
        transition = 1 # moving down or right

    if trial_type != offpath_probe_code:
        if shift_dimension == 0:
            new_position = [last_position[shift_dimension] + transition, last_position[~shift_dimension]]
        elif shift_dimension == 1:
            new_position = [last_position[~shift_dimension], last_position[shift_dimension] + transition]

        if (new_position[0] >= int_reward_graph.shape[0]) or (new_position[1] >= int_reward_graph.shape[1]) or (new_position[0] < 0) or (new_position[1] < 0): # if outside bounds of graph (wall hit), don't increment
            new_position = last_position

    if trial_type == normal_trial_code:
        reward = int_reward_graph[new_position[0], new_position[1]] # find reward in the landscape
    else:
        reward = -999

    if reward == -3: # hit wall
        new_position = coordinate_choice_list[-2] # position prior to hitting wall (wall counts as last pos.)

    coordinate_choice_list.append(new_position)


    if reward != -3: # if not wall
        int_reward_graph[new_position[0], new_position[1]] = 0 # remove reward consumed for future trials. now same behavior as wall.

    if choice == red_key:
        chosen_cue = red_box
        action_choice_list.append("U") # up
        button_choice_list.append("R")
        choice_emphasis.setPos(red_pos)
        rewardMsg.setPos([red_pos[0], red_pos[1]+210])
    elif choice == green_key:
        chosen_cue = green_box
        action_choice_list.append("D") # down
        button_choice_list.append("G")
        choice_emphasis.setPos(green_pos)
        rewardMsg.setPos([green_pos[0], green_pos[1]+210])

    elif choice == blue_key:
        chosen_cue = blue_box
        action_choice_list.append("L") # left
        button_choice_list.append("B")
        choice_emphasis.setPos(blue_pos)
        rewardMsg.setPos([blue_pos[0], blue_pos[1]+210])
    elif choice == yellow_key:
        chosen_cue = yellow_box
        action_choice_list.append("R") # right
        button_choice_list.append("Y")
        choice_emphasis.setPos(yellow_pos)
        rewardMsg.setPos([yellow_pos[0], yellow_pos[1]+210])

    # print('reward ', reward, 'last pos: ', last_position, 'shift dim ', shift_dimension, 'action ', action_choice_list[-1], 'button ', button_choice_list[-1])

    if rt < rt_max and rt > rt_min:

        rewardMsg.setAutoDraw(False)
        window.flip()

        if reward == -3: # hit a wall
            chosen_cue.setAutoDraw(False) # remove cue
            rewardMsg.setAutoDraw(False)
            sum_reward = False
            window.flip()
        if reward == -999: # probe
            rewardMsg.setAutoDraw(False)
            sum_reward = False
            window.flip()
        if reward == 1:
            coin.setPos([chosen_cue.pos[0], chosen_cue.pos[1] + 210])
            coin.setAutoDraw(True)
            # rewardMsg.setAutoDraw(True)
            sum_reward = True
            window.flip()
        if reward == 0:
            rewardMsg.text = str(reward)
            rewardMsg.setAutoDraw(True)
            sum_reward = True
            window.flip()
        if sum_reward:
            total_reward += reward
            cum_reward += reward
            totalMsg.text = str(total_reward)
            totalMsg.setAutoDraw(True)
        cum_reward_list.append(cum_reward)
        received_rewards.append(reward)
        choice_emphasis.draw()
        window.flip()
        coin.setAutoDraw(False)
        rewardMsg.setAutoDraw(False)
        core.wait(fb_time)
        window.flip()

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
    trial_type_list.append(trial_type)
    trialTime_clock.reset()
    print('trial ', t, 'trial_type ', trial_type, 'trial_type_list len ', len(trial_type_list))

    t+=1
    block_trial+=1
    block_n_list.append(block_n)
    trial_list.append(t)

total_exp_time=expTime_clock.getTime()


# save data
print(len(button_choice_list), len(action_choice_list), len(coordinate_choice_list[1:]), len(received_rewards), len(total_rewards), len(rt_list), len(trial_time), len(iti_list),
len(block_n_list), len(trial_list), len(trial_type_list))

if len(button_choice_list) == len(action_choice_list) == len(coordinate_choice_list[1:]) == len(received_rewards) == len(total_rewards) == len(rt_list) == len(trial_time) == len(iti_list) == len(block_n_list) == len(trial_list) == len(trial_type_list):

    data_di = {'button_choice': button_choice_list,
     'action_choice': action_choice_list,'coordinate_choice': coordinate_choice_list[1:], # indexing bc start point included, making it (t+1,1)
    'reward': received_rewards, 'cumulative_reward': total_rewards, 'rt': rt_list, 'total_trial_time': trial_time,
    'iti': iti_list, 'block_n': block_n_list, 'trial_n': trial_list, 'trial_type': trial_type_list}
else:
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
