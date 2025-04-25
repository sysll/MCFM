import matplotlib.pyplot as plt
from PIL import Image

# 图片路径列表
image_paths = [
    "/mnt/dataset/train/腹围/曾凤英_001_175303.jpg",
    "/mnt/dataset/train/脊柱/毕琼婷_001_125811.jpg",
    "/mnt/dataset/train/双肾横断/安云_001_110827.jpg",
    # "/mnt/dataset/train/双肾矢状/阿色木沙作_001_172453.jpg"
]

# 遍历图片路径列表
for image_path in image_paths:
    try:
        # 打开图片
        image = Image.open(image_path)

        # 获取图片尺寸
        width, height = image.size

        # 定义裁剪区域（这里假设裁剪中心部分）
        left = width // 9
        top = height // 8
        right = 8 * width // 9
        bottom = 8 * height // 9

        # 裁剪图片
        cropped_image = image.crop((left, top, right, bottom))

        # 创建一个包含两个子图的画布
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # 显示原始图片
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 显示裁剪后的图片
        axes[1].imshow(cropped_image)
        axes[1].set_title('Cropped Image')
        axes[1].axis('off')

        # 显示图片
        plt.show()

    except FileNotFoundError:
        print(f"错误: 文件 {image_path} 未找到!")
    except Exception as e:
        print(f"错误: 处理文件 {image_path} 时发生未知错误: {e}")
