# TensorFlow 2.0 Hackathon

⚡ #PoweredByTF 2.0 Challenge! [ Devpost ]

<div align="center">
    <b>
        <a href="https://inishchith.github.io/tf-hack2.0/Web/">
            live demo
        </a> 
    </b>
    <br> 
    <img src ="./tfhackdemo.gif" height=550px width=300px alt="demo-gif">
    <p> <b> MoboView </b><p>
</div>


## Story
A lot of countries have started segregating garbage into recyclable and non recyclable. But many times, a lot of people won't be familiar with the items that can be recycled or not. It is beneficial to remember such information but not everyone can do that. To help that, we present a web application which tells you if an object is recyclable or not. But what is special about this? You don't have to put in name of things and search for items. You just click a photo of the object and the app will tell you which garbage bin to throw it into.

## Development
The front end is made is javascript. This then calls a flask app running in the backend which does preprocessing. The flask app then calls the served model. The architecture used is VGG16. Although it's very outdated but it gives a nice beginning to people who are new to Tensorflow-2.0.

PS: Do checkout our video and website. Star us on GitHub and let us know if you have questions/ideas to share!

### How to run

#### Model
See the model/ dir for the README

#### App
Run the app using 
```python3 app.py```
Make sure to set the correct ports and IP addresses for the tensorflow served model and flask app

#### Web
This can be run directly. In app.js make sure the api calls the flask app IP address
