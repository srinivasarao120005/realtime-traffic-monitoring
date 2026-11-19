import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

VEHICLES={"car","motorcycle","bus","truck","bicycle"}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="0",help="Video path, webcam index, or authorized stream URL")
    parser.add_argument("--model",default="yolo11n.pt")
    parser.add_argument("--output",default="runs/traffic/annotated.mp4")
    parser.add_argument("--conf",type=float,default=.30)
    args=parser.parse_args()
    source=int(args.source) if args.source.isdigit() else args.source
    model=YOLO(args.model); cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f"Could not open source: {args.source}")
    writer=None; output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            result=model.predict(frame,conf=args.conf,verbose=False)[0]
            annotated=result.plot(); count=0
            if result.boxes is not None:
                count=sum(model.names[int(c)] in VEHICLES for c in result.boxes.cls.tolist())
            cv2.putText(annotated,f"Vehicles in frame: {count}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            if writer is None:
                h,w=annotated.shape[:2]; fps=cap.get(cv2.CAP_PROP_FPS) or 25
                writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
            writer.write(annotated)
            cv2.imshow("Traffic Monitoring - press q to quit",annotated)
            if cv2.waitKey(1)&0xFF==ord("q"): break
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
    print(f"Annotated video saved to {output}")
if __name__=="__main__": main()
