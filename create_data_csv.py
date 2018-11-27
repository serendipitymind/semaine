import os
import pandas as pd
import soundfile as sf
import numpy as np
import sys
import math
log_file_name = "message.log"
old_stdout = sys.stdout
log_file = open(log_file_name, "w")
sys.stdout = log_file
print("This is log message")
sys.stdout = old_stdout
log_file.close()


def send_message_to_log(s):
    old_stdout = sys.stdout
    log_file = open(log_file_name, "a")
    sys.stdout = log_file
    print(s)
    sys.stdout = old_stdout
    log_file.close()


main_path = r'E:\database\semaine-database\Sessions'
audio_save_path = os.path.join(os.path.dirname(main_path), 'audio')
if not os.path.exists(audio_save_path):
    os.mkdir(audio_save_path)
session_path = [os.path.join(main_path, '{}'.format(i)) for i in range(1, 142)]
data = []
for session in session_path:
    alignedTranscript_files_list = [os.path.join(session, f) for f in os.listdir(session) if f.startswith('wordLevel_alignedTranscript') and f.endswith('user')]
    if alignedTranscript_files_list == [] or len(alignedTranscript_files_list) > 1:
        send_message_to_log("session : {} is unknown".format(session))
        continue
    alignedTranscript_files = alignedTranscript_files_list[0]
    with open(alignedTranscript_files) as file:
        context = file.readlines()
        index_start = [i for i in range(len(context)) if context[i].startswith('---recording')]
        index_end = [i for i in range(len(context)) if context[i] == '.\n']
        assert len(index_start) == len(index_end)
        turn_num = len(index_end)
        turns = [context[start: end+1] for start, end in zip(index_start, index_end) ]
        import re
        regex = re.compile('.*_User HeadMounted.*.wav')
        audio_file_name = [os.path.join(session, item) for item in os.listdir(session) if regex.match(item)][0]
        regex = re.compile('.*U.*DA.txt')
        a_files = [os.path.join(session, item) for item in os.listdir(session) if regex.match(item)]
        regex = re.compile('.*U.*DV.txt')
        v_files = [os.path.join(session, item) for item in os.listdir(session) if regex.match(item)]
        a_f_data = []
        for a_f in a_files:
            with open(a_f, 'r') as f:
                a_data = np.array([sfd.split(' ')[1] for sfd in f.readlines() if not sfd == '\n'], dtype=np.float32)
                a_f_data.append(a_data)
        print(session)
        if session == r'E:\database\semaine-database\Sessions\25':
            print(session)
        if not min([len(a) for a in a_f_data]) == max([len(a) for a in a_f_data]):
            send_message_to_log("{} *DA.txt not equal len".format(session))
        min_len_a = min([len(a) for a in a_f_data])
        max_len_a = max([len(a) for a in a_f_data])
        if min_len_a < max_len_a :
            append_mean_a = np.array([it[min_len_a:] for it in a_f_data if len(it) > min_len_a])
            append_mean_a = append_mean_a.mean(axis=0)
        a_f_data = [a[:min_len_a] for a in a_f_data]
        a_f_data = np.array(a_f_data).mean(axis=0)
        if min_len_a < max_len_a:
            a_f_data = np.append(a_f_data, append_mean_a, axis=-1)
        # except ValueError:
        #     print(" error in {}".format(session))
        v_f_data = []
        for v_f in v_files:
            with open(v_f, 'r') as f:
                v_data = np.array([sfd.split(' ')[1] for sfd in f.readlines() if not sfd == '\n'], dtype=np.float32)
                v_f_data.append(v_data)
        # try:
        if not min([len(a) for a in v_f_data]) == max([len(a) for a in v_f_data]):
            send_message_to_log("{} *DV.txt not equal len".format(session))
        min_len_v = min([len(a) for a in v_f_data])
        max_len_v = max([len(a) for a in v_f_data])
        if min_len_v < max_len_v:
            append_mean_v = np.array([it for it in v_f_data if len(it) > min_len_v])
            append_mean_v = append_mean_v.mean(axis=0)
        v_f_data = [a[:min_len_v] for a in v_f_data]
        v_f_data = np.array(v_f_data).mean(axis=0)
        if min_len_v < max_len_v:
            v_f_data = np.append(v_f_data, append_mean_v, axis=-1)
        # except ValueError:
        #     print(" error in {}".format(session))
        audio_data, sr = sf.read(audio_file_name)
        time = len(audio_data)/sr
        for ind, one_turn in enumerate(turns):
            if len(one_turn) < 3:
                send_message_to_log("session : {0} turn {1} is empty".format(session, ind + 1))
                continue
            # if session == r'E:\database\semaine-database\Sessions\20' and ind == 19:
            # print(session, ind)
            start_time = int(one_turn[1].split(' ')[0])/1000
            end_time = int(one_turn[-2].split(' ')[1])/1000
            lb_start = int(one_turn[1].split(' ')[0])//20 -1 if int(one_turn[1].split(' ')[0])//20 -1 >=0 else 0
            lb_end = math.floor(int(one_turn[-2].split(' ')[1])/20)-1           # -1 扔掉了最后一个，有的标签不够长
            if end_time > time:
                send_message_to_log("session : {0} turn {1} is empty, audio file smaller".format(session, ind + 1))
                continue
            if lb_end > len(v_f_data) or lb_end > len(a_f_data):
                send_message_to_log("session : {0}, turn {1}, label file is smaller than alignedTranscript_files".format(session, ind + 1))
                continue
            st = int(start_time * sr)
            ed = int(end_time * sr)
            seg_audio_data = audio_data[st:ed]
            seg_a = np.array(a_f_data[lb_start: lb_end]).mean(axis=-1)
            seg_v = np.array(v_f_data[lb_start: lb_end]).mean(axis=-1)
            temp = os.path.splitext(os.path.basename(audio_file_name))[0].replace('.', '_').replace(' ', '_')
            seg_audio_name = os.path.join(audio_save_path, 'Ses{0}_{1}_{2:03d}.wav'.format(os.path.basename(session), temp, ind+1))
            sf.write(seg_audio_name, seg_audio_data, sr)
            data_info = []
            data_info.append(os.path.basename(seg_audio_name))
            data_info.append(seg_a)
            data_info.append(seg_v)
            data_info.append((start_time, end_time))
            data_info.append(end_time-start_time)
            data.append(data_info)
        pass
columns = ['audio_name', 'a', 'v', 'start, end time', 'time']
df = pd.DataFrame(data, columns=columns)
df = df.sort_values(by='time', ascending=False)
if not os.path.exists('data'):
    os.mkdir('data')
df.to_csv(os.path.join('data', 'data_file.csv'), index=False)


