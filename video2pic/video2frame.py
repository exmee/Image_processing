import cv2

def video2frame(videos_path, frames_save_path, time_interval):
    '''
    :param videos_path: 视频的存放路径
    :param frames_save_path: 视频切分成帧之后图片的保存路径
    :param time_interval: 保存间隔
    :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
        # if count == 20:
        #   break
    print(count)


if __name__ == '__main__':
    videos_path = 'D:\\sz4361\\公司项目文件\\舱门检测\\舱门视频\\舱门视频\\20191109_161552.mp4'
    frames_save_path = 'D:\\sz4361\\公司项目文件\\舱门检测\\舱门视频\\舱门视频\\20191109_161552'
    time_interval = 2  # 隔一帧保存一次
    video2frame(videos_path, frames_save_path, time_interval)