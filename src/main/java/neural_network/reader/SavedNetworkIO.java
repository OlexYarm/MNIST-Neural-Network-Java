package neural_network.reader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import neural_network.service.NeuralNetwork;

public class SavedNetworkIO {

    private final static String SAVE_DIRECTORY_NAME = "saved_neuron_values";

    public static void createNeuralNetworkSaves(NeuralNetwork neuralNetwork, String neuralNetworkName)
            throws IOException {
        String filePath = SAVE_DIRECTORY_NAME + "/" + neuralNetworkName + ".dat";

        File logDirectory = new File(SAVE_DIRECTORY_NAME);
        if (!logDirectory.exists()) {
            logDirectory.mkdirs();
        }
        try (ObjectOutputStream neuralNetworkOutputStream = new ObjectOutputStream(new FileOutputStream(filePath))) {
            neuralNetworkOutputStream.writeObject(neuralNetwork);
            System.out.println(filePath + " created successfully!");
        } catch (IOException e) {
            System.err.println("Error creating layer log file: " + e.getMessage());
        }
    }

    public static NeuralNetwork initNeuralNetworkFromSaves(NeuralNetwork neuralNetwork,
            String neuralNetworkName)
            throws IOException, ClassNotFoundException {

        String filePath = SAVE_DIRECTORY_NAME + "/" + neuralNetworkName + ".dat";

        try (ObjectInputStream neuralNetworkInputStream = new ObjectInputStream(new FileInputStream(filePath))) {
            neuralNetwork = (NeuralNetwork) neuralNetworkInputStream.readObject();
            System.out.println(filePath + " data read successfully!");
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error reading layer log file: " + e.getMessage());
            System.out.println("Please Train Network");
        }

        return neuralNetwork;
    }
}
