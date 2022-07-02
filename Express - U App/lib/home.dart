import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite/tflite.dart';
import 'dart:math' as math;

import 'camera.dart';
import 'models.dart';

class HomePage extends StatefulWidget {
  final List<CameraDescription> cameras;

  HomePage(this.cameras);

  @override
  _HomePageState createState() => new _HomePageState();
}

class _HomePageState extends State<HomePage> {
  List<dynamic> _recognitions;
  int _imageHeight = 0;
  int _imageWidth = 0;
  String _model = "";

  @override
  void initState() {
    super.initState();
  }

  loadModel() async {
    String res;

    res = await Tflite.loadModel(
            model: "assets/detect.tflite", labels: "assets/labelmap.txt");


    print(res);
  }

  onSelect(model) {
    setState(() {
      _model = model;
    });
    loadModel();
  }

  setRecognitions(recognitions, imageHeight, imageWidth) {
    setState(() {
      _recognitions = recognitions;
      _imageHeight = imageHeight;
      _imageWidth = imageWidth;
    });
  }

  @override
  Widget build(BuildContext context) {
    Size screen = MediaQuery.of(context).size;
    return Scaffold(
      body: _model == ""
          ? Container(
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: AssetImage("assets/bg.png"),
                  fit: BoxFit.cover,
                ),
              ),
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    Image(
                      image: AssetImage("assets/logo.png"),
                    ),
                    ElevatedButton(
                      child: const Text("Start recognition"),
                      onPressed: () => onSelect(detect),
                    ),
                  ],
                ),
              ),
            )
          : Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Container(
                  height: MediaQuery.of(context).size.height * 0.5,
                  width: MediaQuery.of(context).size.width,
                  child: Camera(
                    widget.cameras,
                    _model,
                    setRecognitions,
                  ),
                ),
                SizedBox(
                  height: MediaQuery.of(context).size.height * 0.1,
                ),
                Container(
                  decoration: BoxDecoration(
                    image: DecorationImage(
                      image: AssetImage("assets/bg.png"),
                      fit: BoxFit.fill,
                    ),
                  ),
                  height: MediaQuery.of(context).size.height * 0.3,
                  child: Text(
                    "$_recognitions ",
                    style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.blueGrey),
                  ),
                ),
                Center(
                  child: Text(
                    '© 2022 Express-U. All rights reserved',
                    style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue),
                  ),
                )
              ],
            ),
    );
  }
}
