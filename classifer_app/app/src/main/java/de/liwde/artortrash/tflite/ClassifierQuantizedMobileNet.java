/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package de.liwde.artortrash.tflite;

import android.app.Activity;
import java.io.IOException;

/** This TensorFlow Lite classifier works with the quantized MobileNet model. */
public class ClassifierQuantizedMobileNet extends Classifier {

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private byte[][] probArray = null;

  /**
   * Initializes a {@code ClassifierQuantizedMobileNet}.
   *
   * @param activity
   */
  public ClassifierQuantizedMobileNet(Activity activity, Device device, int numThreads)
      throws IOException {
    super(activity, device, numThreads);
    probArray = new byte[1][1];
  }

  @Override
  public int getImageSizeX() {
    return 224;
  }

  @Override
  public int getImageSizeY() {
    return 224;
  }

  @Override
  protected String getModelPath() {
    // you can download this file from
    // see build.gradle for where to obtain this file. It should be auto
    // downloaded into assets.
    return "model.quant.tflite";
  }

  @Override
  protected int getNumBytesPerChannel() {
    // the quantized model uses a single byte only
    return 4;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
    imgData.put((byte) (pixelValue & 0xFF));
  }

  @Override
  protected float getProbability() {
    return probArray[0][0];
  }

  @Override
  protected void setProbability(Number value) {
    probArray[0][0] = value.byteValue();
  }

  @Override
  protected float getNormalizedProbability() {
    return (probArray[0][0] & 0xff) / 255.0f;
  }

  @Override
  protected void runInference() {
    tflite.run(imgData, probArray);
  }
}
