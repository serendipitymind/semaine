import pandas as pd
import os
import shutil
import re
df = pd.read_csv(os.path.join('data', 'data_file.csv'))
a = df[df['time'] > 6.0]['audio_name'].map(lambda x: x.split('_')[0][3:])
c = set(a)
main_path = r'E:\database\semaine-database\Sessions'
session_path = [os.path.join(main_path, '{}'.format(i)) for i in c]
for session in session_path:
    targetDir = os.path.join(os.path.dirname(main_path), 'qiefen', os.path.basename(session))
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    regex = re.compile('.*_User HeadMounted.*.wav')
    audio_file_name = [os.path.join(session, item) for item in os.listdir(session) if regex.match(item)][0]
    alignedTranscript_files_list = [os.path.join(session, f) for f in os.listdir(session) if f.startswith('wordLevel_alignedTranscript') and f.endswith('user')]
    alignedTranscript_file = alignedTranscript_files_list[0]
    alignedTranscript_sentence = [os.path.join(session, f) for f in os.listdir(session) if f.startswith('alignedTranscript')][0]
    shutil.copy(alignedTranscript_file, targetDir)
    shutil.copy(audio_file_name, targetDir)
    shutil.copy(alignedTranscript_sentence, targetDir)
print(c)