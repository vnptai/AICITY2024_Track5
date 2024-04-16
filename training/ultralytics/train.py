from ultralytics import YOLO

###########################
model = YOLO('./config/yolov8x-p6').load("yolov8x-oiv7.pt")
model.train(data='./config/data_v1.yaml', device="0,1,2,3,4,5,6,7", workers=16, epochs=120, imgsz=1280, batch=16,
            close_mosaic=15,
            project="../weights/yolov8x-p6_data_v1/")

model = YOLO('./config/yolov8x-p2.yaml').load("yolov8x-oiv7.pt")
model.train(data='./config/data_v1.yaml', device="0,1,2,3,4,5,6,7", workers=16, epochs=120, imgsz=1280, batch=16,
            close_mosaic=15,
            project="../weights/yolov8x-p2_data_v1/")

model = YOLO('./config/yolov8x.yaml').load("yolov8x-oiv7.pt")
model.train(data='./config/data_v1.yaml', device="0,1,2,3,4,5,6,7", workers=16, epochs=240, imgsz=832, batch=16,
            close_mosaic=15,
            project="../weights/yolov8x_data_v1/", multi_scale=True)

model = YOLO('./config/yolov8x.yaml').load("yolov8x-oiv7.pt")
model.train(data='./config/data_v2.yaml', device="0,1,2,3,4,5,6,7", workers=16, epochs=240, imgsz=832, batch=16,
            close_mosaic=15,
            project="../weights/yolov8x_data_v2_1/", multi_scale=True)

model = YOLO('./config/yolov8x.yaml').load("yolov8x.pt")
model.train(data='./config/data_v2.yaml',
            epochs=150,
            device="0,1,2,3,4,5,6,7",
            cls=1.0,
            batch=32,
            imgsz=960,
            mosaic=0.2,
            albumentations=0.5,
            shear=5,
            degrees=5,
            close_mosaic=0,
            perspective=0.0005,
            scale=0.5,
            cos_lr=True, project="../weights/yolov8x_data_v2_2/")
