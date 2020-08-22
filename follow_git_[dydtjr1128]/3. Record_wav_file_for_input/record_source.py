import pyaudio  # 마이크를 사용하기 위한 라이브러리
import wave
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 5  # 녹음할 시간 설정
# WAVE_OUTPUT_FILENAME = "record.wav"


def record(who):
    print("record start!")
    # 파일명 조심 : 파일명에 콜론 들어가면 안됨
    now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-')
    WAVE_OUTPUT_FILENAME = "data_record/" + now + str(who) + ".wav"
    print(WAVE_OUTPUT_FILENAME)

    p = pyaudio.PyAudio()  # 오디오 객체 생성

    stream = p.open(format=FORMAT,  # 16비트 포맷
                    channels=CHANNELS, #  모노로 마이크 열기
                    rate=RATE, #비트레이트
                    input=True,
                    # input_device_index=1,
                    frames_per_buffer=CHUNK)
                      # CHUNK만큼 버퍼가 쌓인다.

    print("Start to record the audio.")

    frames = []  # 음성 데이터를 채우는 공간

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        #지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording is finished.")

    stream.stop_stream() # 스트림닫기
    stream.close() # 스트림 종료
    p.terminate() # 오디오객체 종료

    # WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


    # 우선, wave plot 생략
    # spf = wave.open(WAVE_OUTPUT_FILENAME,'r')
    #
    # signal = spf.readframes(-1)
    # signal = np.fromstring(signal, dtype=np.int16)
    #
    # #시간 흐름에 따른 그래프를 그리기 위한 부분
    # Time = np.linspace(0,len(signal)/RATE, num=len(signal))
    #
    # fig1 = plt.figure()
    # plt.title('Voice Signal Wave...')
    # #plt.plot(signal) // 음성 데이터의 그래프
    # plt.plot(Time, signal)
    # plt.show()
    # plt.close(fig1)  # 닫아줘야하는 번거로움
    print("record end!!")

    return WAVE_OUTPUT_FILENAME

