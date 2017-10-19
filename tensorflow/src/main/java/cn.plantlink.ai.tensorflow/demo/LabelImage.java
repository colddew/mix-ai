package cn.plantlink.ai.tensorflow.demo;

import org.tensorflow.*;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
public class LabelImage {

  public static void main(String[] args) {

    byte[] graphDef = readAllBytesOrExit(Paths.get("/tmp/output_graph.pb"));
    List<String> labels = readAllLinesOrExit(Paths.get("/tmp/output_labels.txt"));
    byte[] imageBytes = readAllBytesOrExit(Paths.get("/Users/anmy/Downloads/pic_temp/1.jpg"));

    try (Tensor image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {

      float[] labelProbabilities = executeInceptionGraph(graphDef, image);
      int bestLabelIdx = maxIndex(labelProbabilities);
      System.out.println(String.format("BEST MATCH: %s (%.2f%% likely)", labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));

      Map map = sortIndex(labelProbabilities);
      System.out.println(map);
      map.forEach((k, v) -> {
        System.out.println(k + " -> " + v);
      });
    }
  }

  private static Tensor constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
    try (Graph g = new Graph()) {
      GraphBuilder b = new GraphBuilder(g);
      // - The model was trained with images scaled to 299x299 pixels.
      // - The colors, represented as R, G, B in 1-byte each were converted to
      //   float using (value - Mean)/Scale.
      final int inputHeight = 299;
      final int inputWidth = 299;
      final float mean = 0;
//      final float mean = 128.0f;
//      final float scale = 1f;
      final float inputStd = 255f;

      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
      final Output input = b.constant("input", imageBytes);
      final Output output =
          b.div(
              b.sub(
                  b.resizeBilinear(
                      b.expandDims(
                          b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
                          b.constant("make_batch", 0)),
                      b.constant("size", new int[] {inputHeight, inputWidth})),
                  b.constant("mean", mean)),
              b.constant("std", inputStd));
      try (Session s = new Session(g)) {
        return s.runner().fetch(output.op().name()).run().get(0);
      }
    }
  }

  private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
    String input_layer_name = "Mul";
    String output_layer_name = "final_result";
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);
      try (Session s = new Session(g);
          Tensor result = s.runner().feed(input_layer_name, image).fetch(output_layer_name).run().get(0)) {
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
          throw new RuntimeException(
              String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                  Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[1][nlabels])[0];
      }
    }
  }

  private static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

  private static Map sortIndex(float[] probabilities) {
    Map sortIndexMap = new TreeMap(Collections.reverseOrder());
    for (int i = 1; i < probabilities.length; ++i) {
      sortIndexMap.put(probabilities[i], i);
    }
    return sortIndexMap;
  }

  private static byte[] readAllBytesOrExit(Path path) {
    try {
      return Files.readAllBytes(path);
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(1);
    }
    return null;
  }

  private static List<String> readAllLinesOrExit(Path path) {
    try {
      return Files.readAllLines(path, Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(0);
    }
    return null;
  }

  // In the fullness of time, equivalents of the methods of this class should be auto-generated from
  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
  // like Python, C++ and Go.
  static class GraphBuilder {
    GraphBuilder(Graph g) {
      this.g = g;
    }

    Output div(Output x, Output y) {
      return binaryOp("Div", x, y);
    }

    Output sub(Output x, Output y) {
      return binaryOp("Sub", x, y);
    }

    Output resizeBilinear(Output images, Output size) {
      return binaryOp("ResizeBilinear", images, size);
    }

    Output expandDims(Output input, Output dim) {
      return binaryOp("ExpandDims", input, dim);
    }

    Output cast(Output value, DataType dtype) {
      return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
    }

    Output decodeJpeg(Output contents, long channels) {
      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
          .addInput(contents)
          .setAttr("channels", channels)
          .build()
          .output(0);
    }

    Output constant(String name, Object value) {
      try (Tensor t = Tensor.create(value)) {
        return g.opBuilder("Const", name)
            .setAttr("dtype", t.dataType())
            .setAttr("value", t)
            .build()
            .output(0);
      }
    }

    private Output binaryOp(String type, Output in1, Output in2) {
      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
    }

    private Graph g;
  }
}