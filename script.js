const xProp = 'Population (CDP)';
const yProp = 'Scope-1 GHG emissions [tCO2 or tCO2-eq]'
let status;
let interact;
let model;
let result;
let submitBtn;
let tensorData;

async function getData() {
    let output = await fetch('input/D_FINAL-Selected.json')
    output = await output.json()

    // clean up
    output.data = output.data.filter(item => {
        if (item[xProp] !== "" && item[xProp] !== null && item[xProp] !== undefined && item[xProp] > 1 &&
            item[yProp] !== "" && item[yProp] !== null && item[yProp] !== undefined && item[yProp] > 1) {
            item[xProp] = Math.log10(item[xProp])
            item[yProp] = Math.log10(item[yProp])
            // item[xProp] = parseFloat(item[xProp])
            // item[yProp] = parseFloat(item[yProp])
            return item
        }
    })

    return output
}

function visualizeVals(data) {
    const values = data.data.map(d => ({
        x: d[xProp],
        y: d[yProp],
    }));

    console.log(values)
    tfvis.render.scatterplot({
        name: `${xProp} vs ${yProp}`
    }, {
        values
    }, {
        xLabel: xProp,
        yLabel: yProp,
        height: 400
    });
}


function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single hidden layer
    model.add(tf.layers.dense({
        inputShape: [1],
        units: 1,
        useBias: true
    }));

    // Add an output layer
    model.add(tf.layers.dense({
        units: 1,
        useBias: true
    }));

    return model;
}

function convertToTensor(data) {
    return tf.tidy(() => {
        // Step 1. Shuffle the data    
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d[xProp])
        const labels = data.map(d => d[yProp]);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }

    })
}



async function trainModel(model, inputs, labels) {

    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    })

    const batchSize = 28;
    const epochs = 120;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks({
                name: 'training performance'
            },
            ['loss', 'mse'], {
                height: 300,
                callbacks: ['onEpochEnd']
            }
        )
    })

}


function predict(_val) {
    let convertedInput = tf.tensor([Math.log10(_val)]);
    // let convertedInput = _val;

    const {
        inputMax,
        inputMin,
        labelMin,
        labelMax
    } = tensorData;

    const normalizedInputs = convertedInput.sub(inputMin).div(inputMax.sub(inputMin));

    normalizedInputs.print()
    const [preds] = tf.tidy(() => {
        const preds = model.predict(normalizedInputs)

        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        return [unNormPreds.dataSync()]
    })

    const predictedPoints = Array.from(preds).map((val, i) => {
        return {
            x: convertedInput.dataSync()[0],
            y: val
        }
    });


    return predictedPoints[0]
}


function testModel(model, inputData, normalizationData) {
    const {
        inputMax,
        inputMin,
        labelMin,
        labelMax
    } = normalizationData;

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]))

        const unNormXs = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        return [unNormXs.dataSync(), unNormPreds.dataSync()]
    })

    const predictedPoints = Array.from(xs).map((val, i) => {
        return {
            x: val,
            y: preds[i]
        }
    });

    const originalPoints = inputData.map(d => ({
        x: d[xProp],
        y: d[yProp],
    }));

    tfvis.render.scatterplot({
        name: `model predictions vs original data`
    }, {
        values: [originalPoints, predictedPoints],
        series: ['original', 'predicted']
    }, {
        xLabel: xProp,
        yLabel: yProp,
        height: 400
    });

}


function handleSubmit() {
    result = document.querySelector("#result");
    const input = document.querySelector('#population-input').value

    const prediction = predict(input);
    result.innerHTML =  Math.pow(10, prediction.y)
}




async function make() {
    // dom stuff
    status = document.querySelector("#status");
    interact = document.querySelector("#interact");
    submitBtn = document.querySelector("#submit-btn")
    submitBtn.addEventListener('click', handleSubmit);

    const data = await getData();


    visualizeVals(data)

    // Create the model
    model = createModel();

    tensorData = convertToTensor(data.data);
    const {
        inputs,
        labels
    } = tensorData;

    await trainModel(model, inputs, labels);

    console.log('done training')

    testModel(model, data.data, tensorData);

    // await model.save('downloads://co2Net.json');
    tfvis.show.modelSummary({
        name: 'Model Summary'
    }, model);

    status = document.querySelector("#status");

    status.innerHTML = "done training! if your model didn't achieve a good fit, refresh and try again!";
    status.innerHTML += `<button class="pa2"><a class="link black" href="/">Hmm, these results are crummy, let's try again</a></button>`
    interact.classList.toggle('hidden');

}
make()