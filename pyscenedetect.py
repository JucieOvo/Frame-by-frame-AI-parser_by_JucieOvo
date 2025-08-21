#PySceneDetect
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

def extract_scenes_pyscenedetect(video_path, output_dir="scenedetect_frames", threshold=30.0):
    """
    使用PySceneDetect检测场景变化并提取关键帧
    
    参数:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出目录，用于保存提取的帧
        threshold (float): 场景检测的敏感度阈值(默认30.0)
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 创建视频管理器
        video_manager = VideoManager([video_path])
        
        # 创建场景管理器
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        # 设置视频管理器
        video_manager.set_downscale_factor()  # 可选的降采样以提高处理速度
        
        # 开始视频管理器和场景检测
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # 获取场景列表
        scene_list = scene_manager.get_scene_list()
        
        # 保存每个场景的第一帧作为关键帧
        save_images(
            scene_list, 
            video_manager, 
            output_dir=output_dir,
            image_name_template='scene_$SCENE_NUMBER',
            num_images=1  # 每个场景只保存第一帧
        )
        
        # 释放视频管理器资源
        video_manager.release()
        
        print(f"PySceneDetect场景检测完成，帧已保存到 {output_dir} 目录")
        print(f"检测到 {len(scene_list)} 个场景")
        
    except Exception as e:
        print(f"PySceneDetect处理出错: {e}")

if __name__ == "__main__":
    video_file = "test.mp4"  # 替换为您的视频文件名
    
    if os.path.exists(video_file):
        extract_scenes_pyscenedetect(video_file)
    else:
        print(f"错误: 视频文件 {video_file} 不存在")