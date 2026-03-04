import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# Cài đặt giao diện trang web
st.set_page_config(page_title="Hệ thống Đếm Xe", page_icon="🚦")
st.title("🚦 Hệ thống Đếm Lưu Lượng Giao Thông")
st.write("Tải video lên để hệ thống tự động nhận diện và đếm số lượng xe.")

# Load Model
@st.cache_resource
def load_model():
    # Nếu bạn dùng mô hình tự train, đổi 'yolov8n.pt' thành tên file của bạn (vd: 'best.pt')
    return YOLO('yolov8n.pt') 

model = load_model()
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

# Khu vực upload file
uploaded_file = st.file_uploader("Chọn video giao thông (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Lưu video tải lên vào một file tạm
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Đọc thông số video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"Đang xử lý video... Tổng số: {total_frames} khung hình.")

    # Khung hiển thị giao diện
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    # Khởi tạo bộ đếm
    vehicle_count = {cls: 0 for cls in vehicle_classes}
    detected_ids = set()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy YOLO Tracking
        results = model.track(frame, persist=True, verbose=False)

        # Vẽ và đếm
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2, obj_id, cls_id, conf) in zip(
                boxes.xyxy[:,0], boxes.xyxy[:,1], boxes.xyxy[:,2], boxes.xyxy[:,3],
                ids, classes, confs
            ):
                if cls_id < len(model.names):
                    label = model.names[cls_id]
                else:
                    continue

                if label not in vehicle_classes:
                    continue

                # Vẽ bounding box (Màu Vàng)
                color = (0, 255, 255) 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Cập nhật số lượng đếm theo ID duy nhất
                if obj_id not in detected_ids:
                    vehicle_count[label] += 1
                    detected_ids.add(obj_id)

        # Vẽ bảng thống kê góc phải màn hình
        total = sum(vehicle_count.values())
        overlay = frame.copy()
        box_w, box_h = 250, 160
        x0, y0 = width - box_w - 20, 20
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f'Total Objects: {total}', (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for i, (cls, cnt) in enumerate(vehicle_count.items()):
            cv2.putText(frame, f'{cls.capitalize()}: {cnt}',
                        (x0 + 10, y0 + 60 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Chuyển đổi hệ màu để hiển thị chuẩn trên web (OpenCV dùng BGR, Web dùng RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Đã cập nhật thành use_container_width để không bị lỗi cảnh báo vàng
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Cập nhật tiến trình
        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    st.success("✅ Hoàn tất xử lý video!")
    
    # Hiển thị tổng kết bằng các thẻ metric đẹp mắt của Streamlit
    st.write("### 📊 Tổng kết lưu lượng:")
    cols = st.columns(4)
    cols[0].metric("🚗 Car", vehicle_count['car'])
    cols[1].metric("🚌 Bus", vehicle_count['bus'])
    cols[2].metric("🚚 Truck", vehicle_count['truck'])
    cols[3].metric("🏍️ Motorcycle", vehicle_count['motorcycle'])