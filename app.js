require('dotenv').config();

const express = require('express')
const app = express()
/* 
const expressws = require('express-ws')(app) */

const cors = require('cors');

app.use(cors({ credentials: true, origin: true }));

var bcrypt = require('bcrypt')

var bodyParser = require('body-parser')
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

var saltRounds = 13;

var cjson = require('csvtojson');

const port = process.env.PORT || 3000;

app.get('/search/:key/:count', async (req, res) => {

    let runPy = new Promise(async function (resolve, reject) {

        const { PythonShell } = require('python-shell')
        let pyshell = new PythonShell('./pyFiles/main.py');

        // sends a message to the Python script via stdin
        pyshell.send(JSON.stringify(
            [req.params.key,
            req.params.count,
            process.env.CONSUMER,
            process.env.CONSUMER_SECRET,
            process.env.ACCESS,
            process.env.ACCESS_SECRET
            ]));

        var local = [];

        await pyshell.on('message', async function (message) {
            await console.log("jshf",message);
            local.push(message);
        });
        
        // end the input stream and allow the process to exit
        pyshell.end(function (err, code, signal) {
            if (err) throw err;
            console.log('The exit code was: ' + code);
            console.log('The exit signal was: ' + signal);
            console.log('finished');
            resolve(local); 
        });
    })
    await runPy.then(data=>{
        res.status(200);
        res.send({message:"success"});
        res.end();
    }).catch(err=>{
        console.log("jshkvfjds",err);
        res.status(409)
        res.send({message:"Error Occured during NLP"})
        res.end();
    })
});

app.get('/analyse', async (req, res) => {

    let runPy = new Promise(async function (resolve, reject) {

        const { PythonShell } = require('python-shell')
        let pyshell = new PythonShell('./pyFiles/main2.py');

        var local = [];

        await pyshell.on('message', async function (message) {
            await console.log("jshf",message);
            local.push(message);
        });
        
        // end the input stream and allow the process to exit
        pyshell.end(function (err, code, signal) {
            if (err) throw err;
            console.log('The exit code was: ' + code);
            console.log('The exit signal was: ' + signal);
            console.log('finished');
            resolve(local); 
        });
    })
    await runPy.then(data=>{
        res.status(200);
        res.send({message:"success"});
        res.end();
    }).catch(err=>{
        console.log("jshkvfjds",err);
        res.status(409);
        res.send({message:"Error Occured during Clustering.."})
        res.end();
    })
});

app.get("/getResult",async (req,res) =>{
    cjson().fromFile('./pyFiles/dataset2.csv')
        .then( (jsonArrayObj) =>{ //when parse finished, result will be emitted here.
            console.log(jsonArrayObj);
            res.send({data:jsonArrayObj}); 
   }).catch(err=>{
       res.status(409)
       res.send({message:"Error during parsing..."})
   })
});

app.listen(port, () => console.log(`Application listening on port ${port}`)) 