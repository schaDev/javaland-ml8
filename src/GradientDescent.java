import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GradientDescent {

    public static void main(String[] args) {
        int[] xData = {1, 2, 3, 4};
        int[] yData = {1, 3, 2, 3};
        gradientDescent(xData, yData);
    }

    private static void gradientDescent(int[] xData, int[] yData) {
        List<Dto> data = IntStream.range(0, xData.length).boxed().map(i -> new Dto(xData[i], yData[i])).collect(Collectors.toList());
        double a = 1;
        double b = 1;
        long epochs = 1000;
        double learningRate = 0.01;
        long n = data.size();

        for (int epoch = 1; epoch < epochs; epoch++) {
            final Dto p = new Dto(a, b);
            ToDoubleFunction<Double> yPredicted = x -> p.a * x + p.b;

            double cost = (1d / n) * data.stream().mapToDouble(d -> d.y - Math.pow(d.y - yPredicted.applyAsDouble(d.x), 2)).sum();

            double da = (1d / n) * data.stream().mapToDouble(d -> 2 * (d.y - yPredicted.applyAsDouble(d.x)) * -d.x).sum();
            double db = (1d / n) * data.stream().mapToDouble(d -> 2 * (d.y - yPredicted.applyAsDouble(d.x)) * -1).sum();

            a = a - learningRate * da;
            b = b - learningRate * db;

            System.out.format("a %f, b %f cost %f epoch %d%n", a, b, cost, epoch);
        }
    }

    private static class Dto {
        double a;
        double b;
        double x;
        double y;

        public Dto(double a, double b) {
            this.a = a;
            this.b = b;
        }

        public Dto(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

}
