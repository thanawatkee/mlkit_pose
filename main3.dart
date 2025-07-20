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

  // ตรวจจับท่านั่ง
  Timer? _sitTimer;
  bool _isSitting = false;
  bool _sitConfirmed = false;

  // ตรวจจับการล้ม
  bool _wasStanding = false;
  DateTime? _lastStandingTime;
  bool _fallDetected = false;

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
      final rotation = InputImageRotationValue.fromRawValue(cameras.first.sensorOrientation) ??
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

        /// ตรวจจับการนั่ง
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

        /// ตรวจจับการล้ม
        if (poseLabel == 'ยืน') {
          _wasStanding = true;
          _lastStandingTime = DateTime.now();
          _fallDetected = false;
        } else if (_wasStanding && poseLabel == 'ล้ม') {
          final fallDuration = DateTime.now().difference(_lastStandingTime ?? DateTime.now());
          if (fallDuration.inSeconds < 3) {
            setState(() {
              _fallDetected = true;
            });
            _wasStanding = false;
          }
        }

        setState(() {
          _poses = poses;
          _currentPoseLabel = poseLabel;
        });
      } else {
        _isSitting = false;
        _sitTimer?.cancel();
        _sitConfirmed = false;
        _fallDetected = false;

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
    final leftHip = pose.landmarks[PoseLandmarkType.leftHip];
    final leftKnee = pose.landmarks[PoseLandmarkType.leftKnee];
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final leftShoulder = pose.landmarks[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose.landmarks[PoseLandmarkType.rightShoulder];
    final leftWrist = pose.landmarks[PoseLandmarkType.leftWrist];

    if (leftHip == null || leftKnee == null || leftAnkle == null) return 'ไม่ทราบท่าทาง';

    final kneeAngle = _calculateAngle(leftHip, leftKnee, leftAnkle);

    // ตรวจจับล้ม: กว้างมากกว่าสูง
    if (leftShoulder != null && rightShoulder != null) {
      final width = (leftShoulder.x - rightShoulder.x).abs();
      final height = (leftShoulder.y - leftHip.y).abs();

      if (width > height * 0.8) {
        return 'ล้ม';
      }
    }

    if (kneeAngle > 70 && kneeAngle < 110) return 'นั่ง';
    if (kneeAngle > 160) return 'ยืน';
    if (leftWrist != null && leftShoulder != null && leftWrist.y < leftShoulder.y) {
      return 'ยกมือ';
    }

    return 'ไม่ทราบท่าทาง';
  }

  double _calculateAngle(PoseLandmark a, PoseLandmark b, PoseLandmark c) {
    final baX = a.x - b.x;
    final baY = a.y - b.y;
    final bcX = c.x - b.x;
    final bcY = c.y - b.y;

    final dot = baX * bcX + baY * bcY;
    final magBA = sqrt(baX * baX + baY * baY);
    final magBC = sqrt(bcX * bcX + bcY * bcY);

    final angleRad = acos(dot / (magBA * magBC + 1e-6));
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
            child: Container(),
          ),
          Positioned(
            top: 20,
            left: 20,
            child: _infoBox('ท่าปัจจุบัน: $_currentPoseLabel', Colors.black),
          ),
          if (_sitConfirmed)
            Positioned(
              top: 70,
              left: 20,
              child: _infoBox('✅ นั่งครบ 5 วินาที', Colors.green),
            ),
          if (_fallDetected)
            Positioned(
              top: 120,
              left: 20,
              child: _infoBox('❗ ตรวจพบว่าผู้ใช้ล้มลง', Colors.red),
            ),
        ],
      ),
    );
  }

  Widget _infoBox(String text, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.85),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: const TextStyle(color: Colors.white, fontSize: 18),
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
  bool shouldRepaint(covariant PosePainter oldDelegate) => true;
}
