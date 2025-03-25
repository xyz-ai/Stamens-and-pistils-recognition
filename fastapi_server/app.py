from fastapi import FastAPI, File, UploadFile, Request
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn
from fastapi.staticfiles import StaticFiles

# 创建 FastAPI 实例
app = FastAPI()

# 加载 YOLO 模型
model = YOLO("runs/detect/train5/weights/best.pt")  # 确保路径正确

# 静态文件（CSS/JS）
STATIC_DIR = Path(__file__).parent / "static"  # 获取当前脚本所在目录的 static 目录
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir()  # 确保目录存在

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 存储上传的文件
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 首页：显示 HTML 界面
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>玉米雄穗识别</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            h1 { color: #333; }
            #result { margin-top: 20px; }
            img { max-width: 100%; height: auto; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>玉米雄穗识别</h1>
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" id="file-input" name="file" accept="image/*" required>
            <button type="submit">上传并检测</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById("upload-form").onsubmit = async function(event) {
                event.preventDefault();
                let formData = new FormData();
                let fileInput = document.getElementById("file-input");
                formData.append("file", fileInput.files[0]);

                let response = await fetch("/detect/", { method: "POST", body: formData });
                let result = await response.json();

                if (result.result_image) {
                    document.getElementById("result").innerHTML = 
                        '<h3>检测结果：</h3><img src="' + result.result_image + '">';
                } else {
                    document.getElementById("result").innerHTML = "<h3>检测失败！</h3>";
                }
            };
        </script>
    </body>
    </html>
    """

# 处理图片上传和检测
@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    """ 接收图片，进行检测，并返回检测后的图片路径 """
    img_path = UPLOAD_DIR / file.filename
    with img_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 读取图片并进行检测
    img = cv2.imread(str(img_path))
    results = model(img)

    # 解析检测结果并绘制
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 目标框
            confidence = float(box.conf[0])  # 置信度
            label = int(box.cls[0])  # 类别
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存检测结果图片
    output_path = UPLOAD_DIR / f"result_{file.filename}"
    cv2.imwrite(str(output_path), img)

    return JSONResponse(content={
        "result_image": f"/files/{output_path.name}"
    })

# 提供检测后的图片
@app.get("/files/{filename}")
async def get_file(filename: str):
    return FileResponse(UPLOAD_DIR / filename)

# 运行服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
