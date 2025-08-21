#ffmpeg
import subprocess
import os

def extract_iframes_ffmpeg(video_path, output_dir="ffmpeg_frames"):
    """
    使用FFmpeg提取视频中的所有I帧
    
    参数:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出目录，用于保存提取的帧
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 构建FFmpeg命令
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', "select='eq(pict_type,I)'",  # 选择I帧
        '-vsync', 'vfr',                    # 可变帧率输出
        os.path.join(output_dir, 'frame_%04d.jpg')
    ]
    
    try:
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"FFmpeg I帧提取完成，帧已保存到 {output_dir} 目录")
            
            # 统计提取的帧数
            frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
            print(f"共提取了 {frame_count} 个I帧")
        else:
            print(f"FFmpeg执行出错: {result.stderr}")
    except Exception as e:
        print(f"执行FFmpeg命令时出错: {e}")

if __name__ == "__main__":
    video_file = "test.mp4"  # 替换为您的视频文件名
    
    if os.path.exists(video_file):
        extract_iframes_ffmpeg(video_file)
    else:
        print(f"错误: 视频文件 {video_file} 不存在")