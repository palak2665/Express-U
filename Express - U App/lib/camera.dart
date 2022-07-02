import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as imageLib;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:logger/logger.dart';
import 'package:tflite/tflite.dart';
import 'dart:math' as math;
import 'package:path/path.dart';
import 'package:async/async.dart';

import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'models.dart';

typedef void Callback(List<dynamic> list, int h, int w);

class Camera extends StatefulWidget {
  final List<CameraDescription> cameras;
  final Callback setRecognitions;
  final String model;

  Camera(this.cameras, this.model, this.setRecognitions);

  @override
  _CameraState createState() => new _CameraState();
}

class _CameraState extends State<Camera> {
  CameraController controller;
  bool isDetecting = false;

  var logger = Logger();

  var array = [];
  int counter = 0;

  @override
  void initState() {
    super.initState();

    if (widget.cameras == null || widget.cameras.length < 1) {
      print('No camera is found');
    } else {
      controller = new CameraController(
        widget.cameras[0],
        ResolutionPreset.high,
      );
      controller.initialize().then((_) {
        if (!mounted) {
          return;
        }
        setState(() {});

        controller.startImageStream((CameraImage img) async {
          print(widget.model);

          if (isDetecting == false) {
            isDetecting = true;

            int startTime = new DateTime.now().millisecondsSinceEpoch;
            var img1 = ImageUtils.convertCameraImage(img);
            ImageUtils imgutl = ImageUtils();
            var img2 = await imgutl.saveImage(img1);

            Tflite.runModelOnFrame(
              bytesList: img.planes.map((plane) {
                return plane.bytes;
              }).toList(),
              imageHeight: img.height,
              imageWidth: img.width,
              numResults: 2,
            ).then((recognitions) async {
              int endTime = new DateTime.now().millisecondsSinceEpoch;
              print("Detection took ${endTime - startTime}");
              var ans = await upload(File(img2));
              widget.setRecognitions(ans, img.height, img.width);

              isDetecting = false;
            });
          }
        });
      });
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller.value.isInitialized) {
      return Container();
    }

    var tmp = MediaQuery.of(context).size;
    var screenH = math.max(tmp.height, tmp.width);
    var screenW = math.min(tmp.height, tmp.width);
    tmp = controller.value.previewSize;
    var previewH = math.max(tmp.height, tmp.width);
    var previewW = math.min(tmp.height, tmp.width);
    var screenRatio = screenH / screenW;
    var previewRatio = previewH / previewW;

    return OverflowBox(
      maxHeight:
          screenRatio > previewRatio ? screenH : screenW / previewW * previewH,
      maxWidth:
          screenRatio > previewRatio ? screenH / previewH * previewW : screenW,
      child: CameraPreview(controller),
    );
  }



  upload(File img) async {
    print("upload called");

    // open a bytestream
    var bytes = await img.readAsBytes();
    print(bytes);
    var f = img.path;

    // logger.w(bytes);

    var stream = new http.ByteStream(DelegatingStream.typed(img.openRead()));
    print("bytes=== $stream");
    // get file length
    var length = await img.length();
    print("length ==== $length");
    // logger.w(length);

    // string to uri
    var uri = Uri.parse("http://192.168.192.118:5000/predict");
    print("/predict called");

    // create multipart request
    var request = new http.MultipartRequest("POST", uri);
    print("request === $request");



    // add file to multipart
    request.files.add(await http.MultipartFile.fromPath('file', img.path));


    // send
    final response = await http.Response.fromStream(await request.send());

    print("response.body");
    logger.w(response.body);

    return response.body;

  }

  saveImage(imageLib.Image image, [int i = 0]) async {
    List<int> jpeg = imageLib.JpegEncoder().encodeImage(image);
    final appDir = await getTemporaryDirectory();
    final appPath = appDir.path;
    final fileOnDevice = File('$appPath/out$counter.jpg');
    await fileOnDevice.writeAsBytes(jpeg, flush: true);
    Uint8List imagebytes = await fileOnDevice.readAsBytes();
    String base64String = base64.encode(imagebytes);
    array.add(base64String);
    print(array.length);
  }
}



class ImageUtils {
  static imageLib.Image convertCameraImage(CameraImage cameraImage) {
    if (cameraImage.format.group == ImageFormatGroup.yuv420) {
      return convertYUV420ToImage(cameraImage);
    } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
      return convertBGRA8888ToImage(cameraImage);
    } else {
      return null;
    }
  }

  static imageLib.Image convertBGRA8888ToImage(CameraImage cameraImage) {
    imageLib.Image img = imageLib.Image.fromBytes(cameraImage.planes[0].width,
        cameraImage.planes[0].height, cameraImage.planes[0].bytes,
        format: imageLib.Format.bgra);
    return img;
  }

  /// Converts a [CameraImage] in YUV420 format to [imageLib.Image] in RGB format
  static imageLib.Image convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel;

    final image = imageLib.Image(width, height);

    for (int w = 0; w < width; w++) {
      for (int h = 0; h < height; h++) {
        final int uvIndex =
            uvPixelStride * (w / 2).floor() + uvRowStride * (h / 2).floor();
        final int index = h * width + w;

        final y = cameraImage.planes[0].bytes[index];
        final u = cameraImage.planes[1].bytes[uvIndex];
        final v = cameraImage.planes[2].bytes[uvIndex];

        image.data[index] = ImageUtils.yuv2rgb(y, u, v);
      }
    }
    return image;
  }

  /// Convert a single YUV pixel to RGB
  static int yuv2rgb(int y, int u, int v) {
    // Convert yuv pixel to rgb
    int r = (y + v * 1436 / 1024 - 179).round();
    int g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
    int b = (y + u * 1814 / 1024 - 227).round();

    // Clipping RGB values to be inside boundaries [ 0 , 255 ]
    r = r.clamp(0, 255);
    g = g.clamp(0, 255);
    b = b.clamp(0, 255);

    return 0xff000000 |
        ((b << 16) & 0xff0000) |
        ((g << 8) & 0xff00) |
        (r & 0xff);
  }

  Future<String> saveImage(imageLib.Image image, [int i = 0]) async {
    List<int> jpeg = imageLib.JpegEncoder().encodeImage(image);
    final appDir = await getTemporaryDirectory();
    final appPath = appDir.path;
    final fileOnDevice = File('$appPath/out$i.jpg');
    await fileOnDevice.writeAsBytes(jpeg, flush: true);

    print('Saved $appPath/out$i.jpg');
    return '$appPath/out$i.jpg';
  }
}
