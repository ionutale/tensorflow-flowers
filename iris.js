// const * as tf = require("@tensorflow/tfjs")
const tf  = require("@tensorflow/tfjs-node")
const iris = require("./models/iris.json")
const irisTesting = require("./models/iris-testing.json")

// convert/setup our data
const trainingData = tf.tensor2d(iris.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
]))
const outputData = tf.tensor2d(iris.map(item => [
  item.species === "setosa" ? 1 : 0,
  item.species === "virginica" ? 1 : 0,
  item.species === "versicolor" ? 1 : 0,
]))
const testingData = tf.tensor2d(irisTesting.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
]))

// build neural network
const model = tf.sequential()

model.add(tf.layers.dense({
  inputShape: [4],
  activation: "sigmoid",
  units: 50,
}))

model.add(tf.layers.dense({
  inputShape: [50],
  activation: "sigmoid",
  units: 30,
}))
model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 3,
}))

model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06),
})
// train/fit our network
const startTime = Date.now()
model.fit(trainingData, outputData, {epochs: 100})
  .then(async (history) => {
    // console.log(history)
    model.predict(testingData).print()
    model.setUserDefinedMetadata({
      "trainingTime": Date.now() - startTime,
      "trainingEpochs": history.epoch,
      "trainingLoss": history.history.loss[0],
      "trainingAccuracy": history.history.acc[0],
      "model": "iris-model",
      name: "iris-model",
    })
    await model.save(`file://${__dirname}/models`, {
      saveWeights: true,
      saveBiases: true,
      includeOptimizer: true,
      
    });

  })
// test network