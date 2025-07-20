import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:permission_handler/permission_handler.dart';

late List<CameraDescription> cameras;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: PoseDetectionScreen(),
    );
  }
}

class PoseDetectionScreen extends StatefulWidget {
  const PoseDetectionScreen({super.key});

  @override
  State<PoseDetectionScreen> createState() => _PoseDetectionScreenState();
}

class _PoseDetectionScreenState extends State<PoseDetectionScreen> {
  late CameraController _cameraController;
  late PoseDetector _poseDetector;
  bool _isProcessing = false;
  List<Pose> _poses = [];

  Timer? _sitTimer;
  bool _isSitting = false;
  bool _sitConfirmed = false;

  String _currentPoseLabel = 'ไม่ทราบท่าทาง';

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await Permission.camera.request();
    if (await Permission.camera.isGranted) {
      _poseDetector = PoseDetector(
        options: PoseDetectorOptions(mode: PoseDetectionMode.stream),
      );
      _startCamera();
    }
  }

  void _startCamera() {
    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    _cameraController.initialize().then((_) {
      if (!mounted) return;

      _cameraController.startImageStream((CameraImage image) {
        if (_isProcessing) return;
        _isProcessing = true;
        _processImage(image);
      });

      setState(() {});
    });
  }

  Future<void> _processImage(CameraImage image) async {
    try {
      final WriteBuffer allBytes = WriteBuffer();
      for (final Plane plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();

      final Size imageSize = Size(image.width.toDouble(), image.height.toDouble());
      final camera = cameras.first;
      final rotation = InputImageRotationValue.fromRawValue(camera.sensorOrientation) ??
          InputImageRotation.rotation0deg;
      final format = InputImageFormatValue.fromRawValue(image.format.raw) ??
          InputImageFormat.nv21;

      final planeData = image.planes.map(
        (Plane plane) => InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        ),
      ).toList();

      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        inputImageData: InputImageData(
          size: imageSize,
          imageRotation: rotation,
          inputImageFormat: format,
          planeData: planeData,
        ),
      );

      final poses = await _poseDetector.processImage(inputImage);

      if (poses.isNotEmpty) {
        final pose = poses.first;
        final poseLabel = _classifyPose(pose);

        if (_checkIfSitting(pose)) {
          if (!_isSitting) {
            _isSitting = true;
            _sitTimer?.cancel();
            _sitTimer = Timer(const Duration(seconds: 5), () {
              setState(() {
                _sitConfirmed = true;
              });
            });
          }
        } else {
          _isSitting = false;
          _sitTimer?.cancel();
          _sitConfirmed = false;
        }

        setState(() {
          _poses = poses;
          _currentPoseLabel = poseLabel;
        });
      } else {
        _isSitting = false;
        _sitTimer?.cancel();
        _sitConfirmed = false;
        setState(() {
          _poses = [];
          _currentPoseLabel = 'ไม่พบผู้ใช้';
        });
      }
    } catch (e) {
      print("Error processing image: $e");
    } finally {
      _isProcessing = false;
    }
  }

  bool _checkIfSitting(Pose pose) {
    final hip = pose.landmarks[PoseLandmarkType.leftHip];
    final knee = pose.landmarks[PoseLandmarkType.leftKnee];
    final ankle = pose.landmarks[PoseLandmarkType.leftAnkle];

    if (hip == null || knee == null || ankle == null) return false;

    final angle = _calculateAngle(hip, knee, ankle);
    return angle > 70 && angle < 110;
  }

  String _classifyPose(Pose pose) {
    final hip = pose.landmarks[PoseLandmarkType.leftHip];
    final knee = pose.landmarks[PoseLandmarkType.leftKnee];
    final ankle = pose.landmarks[PoseLandmarkType.leftAnkle];

    final shoulder = pose.landmarks[PoseLandmarkType.leftShoulder];
    final wrist = pose.landmarks[PoseLandmarkType.leftWrist];
    final elbow = pose.landmarks[PoseLandmarkType.leftElbow];

    if (hip == null || knee == null || ankle == null) return 'ไม่ทราบท่าทาง';

    final kneeAngle = _calculateAngle(hip, knee, ankle);

    if (kneeAngle > 70 && kneeAngle < 110) return 'นั่ง';
    if (kneeAngle > 160) return 'ยืน';
    if (wrist != null && shoulder != null && wrist.y < shoulder.y) return 'ยกมือ';

    return 'ไม่ทราบท่าทาง';
  }

  double _calculateAngle(PoseLandmark a, PoseLandmark b, PoseLandmark c) {
    final baX = a.x - b.x;
    final baY = a.y - b.y;
    final bcX = c.x - b.x;
    final bcY = c.y - b.y;

    final dot = baX * bcX + baY * bcY;
    final magnitudeBA = sqrt(baX * baX + baY * baY);
    final magnitudeBC = sqrt(bcX * bcX + bcY * bcY);

    final angleRad = acos(dot / (magnitudeBA * magnitudeBC + 1e-6));
    return angleRad * (180 / pi);
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _poseDetector.close();
    _sitTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          CustomPaint(
            painter: PosePainter(_poses, _cameraController.value.previewSize!),
          ),
          Positioned(
            top: 20,
            left: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                'ท่าปัจจุบัน: $_currentPoseLabel',
                style: const TextStyle(color: Colors.white, fontSize: 18),
              ),
            ),
          ),
          if (_sitConfirmed)
            Positioned(
              top: 60,
              left: 20,
              child: Container(
                padding: const EdgeInsets.all(12),
                color: Colors.green.withOpacity(0.8),
                child: const Text(
                  '✅ ตรวจสอบแล้ว: นั่งครบ 5 วินาที',
                  style: TextStyle(color: Colors.white, fontSize: 18),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class PosePainter extends CustomPainter {
  final List<Pose> poses;
  final Size imageSize;

  PosePainter(this.poses, this.imageSize);

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / imageSize.height;
    final scaleY = size.height / imageSize.width;

    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 4.0
      ..style = PaintingStyle.fill;

    for (final pose in poses) {
      for (final point in pose.landmarks.values) {
        final x = size.width - (point.x * scaleX);
        final y = point.y * scaleY;
        canvas.drawCircle(Offset(x, y), 5, paint);
      }
    }
  }

  @override
  bool shouldRepaint(PosePainter oldDelegate) => true;
}
