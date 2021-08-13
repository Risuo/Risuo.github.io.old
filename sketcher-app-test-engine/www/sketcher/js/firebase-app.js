var map;
var centerUrl;
var mapZoom;
var static_Img;
var blob;
var bounds;
var overlay;
var subBounds;
var image;
var predictionOverlay;
var predictionBounds;
var count;

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
var firebaseConfig = {
    apiKey: "AIzaSyCiTASyv4ikDvjz3nRgbGNiUAn-Z4MOLlI",
    authDomain: "sketcher-app-test-engine.firebaseapp.com",
    projectId: "sketcher-app-test-engine",
    storageBucket: "sketcher-app-test-engine.appspot.com",
    messagingSenderId: "987456894174",
    appId: "1:987456894174:web:254008b41a47334a207e4f",
    measurementId: "G-XSHW4JDZQ4"

};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);

function appendScript( url, callback ) {
    var script = document.createElement( "script" )
    script.type = "text/javascript";
    if(script.readyState) {  //IE
        script.onreadystatechange = function() {
            if ( script.readyState === "loaded" || script.readyState === "complete" ) {
                script.onreadystatechange = null;
                callback();
            }
        };
    } else {  //Others
        script.onload = function() {
            callback();
        };
    }

    script.src = url;
    document.getElementsByTagName( "head" )[0].appendChild( script );
}


function appendMap() {
    appendScript("https://maps.googleapis.com/maps/api/js?key=AIzaSyCiTASyv4ikDvjz3nRgbGNiUAn-Z4MOLlI&v=3.exp&libraries=places", function() {
            $("#activate-map").css("display", "none");
            initialize();
            btnMLBegin.classList.remove('hide');
    });
}


function initialize() {
    firebase.auth().signInAnonymously(); // Sets a UID upon appendMap (activate-map button click)
    let count = 0;


    var latlng = new google.maps.LatLng(38.89763405057457, -77.03643959073791); // The White House, DC, USA

    var myOptions = {
        zoom: 19,
        center: latlng,
        mapTypeId: 'satellite',
        gestureHandling: 'greedy',
        streetViewControl: false,
        scaleControl: true,
        fullscreenControl: false,
        tilt: 0,
        rotateControl: false, // This line keeps the tilt option from appearing when the viewport is changed
        mapTypeControlOptions: {
        mapTypeIds: ['']
        }
    };

    // add the map to the map placeholder
    map = new google.maps.Map(document.getElementById("map_canvas"),myOptions);
    map.setTilt(0);

    //const coordsDiv = document.getElementById("coords");
    //const toggleDOMButton = document.createElement("button");
    //toggleDOMButton.textContent = "Toggle Prediction Overlay";
    //toggleDOMButton.classList.add("custom-map-control-buttom");
    //map.controls[google.maps.ControlPosition.TOP_RIGHT].push(toggleDOMButton);

    const input = document.getElementById("pac-input");

    const autocomplete = new google.maps.places.Autocomplete(input);
    autocomplete.setFields(['geometry'])

    const infowindow = new google.maps.InfoWindow();
    const infowindowContent = document.getElementById("infowindow-content");
    infowindow.setContent(infowindowContent);
    const marker = new google.maps.Marker({
        map,
        anchorPoint: new google.maps.Point(0, -29),
    });
    autocomplete.addListener("place_changed", () => {
        infowindow.close();
        marker.setVisible(false);
        const place = autocomplete.getPlace();
        console.log(place)

        if (!place.geometry || !place.geometry.location) {
            window.alert("No details available for input: '" + place.name + "'");
            return;
        }

       if (place.geometry.viewport) {
           map.fitBounds(place.geometry.viewport);
           map.setZoom(19);
       } else {
           map.setCenter(place.geometry.location);
           map.setZoom(19);
       }
       marker.setPosition(place.geometry.location);
       marker.setVisible(true);
       infowindow.open(marker);
    });

    bounds = map.getBounds();
    //map.addListener("mousemove", (event) => {
        //coordsDiv.textContent =
            //"Center: " +
            //map.getCenter() +
            //bounds = map.getBounds()
            //console.log(bounds.getNorthEast().lat())
            //" Zoom: " +
            //map.getZoom()
            // +
            //" lat: " +
            //(event.latLng.lat()) +
            //", " +
            //"lng: " +
            //(event.latLng.lng());
    //});
    map.addListener("click", (event) => {
        centerUrl = map.getCenter().toUrlValue();
        mapZoom = map.getZoom();
        bounds = map.getBounds();
    });

    map.addListener("mouseout", (event) => {
        centerUrl = map.getCenter().toUrlValue();
        mapZoom = map.getZoom();
        bounds = map.getBounds();
    });

    map.addListener("dragend", (event) => {
        centerUrl = map.getCenter().toUrlValue();
        mapZoom = map.getZoom();
        bounds = map.getBounds();
    });

    map.addListener("zoom_changed", (event) => {
        centerUrl = map.getCenter().toUrlValue();
        mapZoom = map.getZoom();
        bounds = map.getBounds();
    });

    map.addListener("idle", (event) => {
        centerUrl = map.getCenter().toUrlValue();
        mapZoom = map.getZoom();
        bounds = map.getBounds();
    });


    const btnMLBegin = document.getElementById('btnMLBegin')
    const btnMLShow = document.getElementById('btnMLShow')

// Click button to grab static image & display based on variables on dynamic map & upload to Firebase storage & retrieve URL for static map

    btnMLBegin.addEventListener('click', e => {
        count ++;
        console.log(count)
        console.log(centerUrl, mapZoom);
        predictionBounds = {
            north: bounds.getNorthEast().lat(),
            south: bounds.getSouthWest().lat(),
            east: bounds.getNorthEast().lng(),
            west: bounds.getSouthWest().lng()
        };

        btnMLShow.classList.remove('hide');
        btnMLBegin.classList.add('hide');


        console.log('Bounds of Viewport = Bounds of Static Img:', predictionBounds)
        static_Img = document.createElement('img');
        static_Img.src = 'https://maps.googleapis.com/maps/api/staticmap?center=' + centerUrl + '&zoom=' + mapZoom + '&maptype=satellite' + '&scale=2' + '&size=640x640' +
            '&format=png32' + '&key=AIzaSyCiTASyv4ikDvjz3nRgbGNiUAn-Z4MOLlI&v=3.exp&libraries=places';
        //document.getElementById('staticGrabSpot').appendChild(static_Img);
        var static_Url = static_Img.src;
        console.log('static_Img URL:', static_Url)
        var uid = firebase.auth().currentUser.uid;
        var filename = firebase.storage().ref('satelite_screenshots/' + uid + '_' + count);

        fetch(static_Url).then(res => {
            return res.blob();
        }).then(blob => {
            filename.put(blob).then(function(snapshot) { // actually, might be able to remove after the put(blob) bit.
                return snapshot.ref.getDownloadURL()
            }).then(url => { // This and the next 2 lines, starting at .then ending with }) can be removed once debugging is done
                console.log("Firebase storage image uploaded available at: ", url);
            })
        }).catch(error => {
            console.error(error);
        });
    });


    btnMLShow.addEventListener('click', e => {
        var uid = firebase.auth().currentUser.uid;
        map.setTilt(0);

        btnMLBegin.classList.remove('hide');
        btnMLShow.classList.add('hide');

        predicted_Img = document.createElement('img');

        var storage = firebase.storage();
        var predicted_Img_Path = storage.refFromURL('gs://sketcher-app-test-engine.appspot.com/satelite_screenshots/' + uid + '_test_out_' + count + '.png')

        predicted_Img_Path.getDownloadURL()
            .then((url) => {
                predicted_Img.src = url
                console.log('Predicted_Img uploaded here:', predicted_Img.src)
                predictionOverlay = new google.maps.GroundOverlay(
                    predicted_Img.src, predictionBounds
                );
                predictionOverlay.setMap(map);
            })
    });
}


//showPrediction.addEventListener('click', e => {
//var uid = firebase.auth().currentUser.uid;
//console.log("Click registered")

//predicted_Img = document.createElement('img');

//predicted_Img.src = 'https://storage.cloud.google.com/sketcher-app-test-engine.appspot.com/satelite_screenshots/' + uid + '_test_out_' + count + '.png'

//document.getElementById('staticGrabSpot').appendChild(predicted_Img);

//});

//toggleDOMButton.addEventListener('click', e => {
//    var uid = firebase.auth().currentUser.uid;
//    map.setTilt(0);
//    console.log("Click registered on toggleDom Btn")


//    predicted_Img = document.createElement('img');

//    var storage = firebase.storage();
//    var predicted_Img_Path = storage.refFromURL('gs://sketcher-app-test-engine.appspot.com/satelite_screenshots/' + uid + '_test_out_' + count + '.png')
//    console.log(predicted_Img_Path)

//    predicted_Img_Path.getDownloadURL()
//        .then((url) => {
//            predicted_Img.src = url
//            console.log('testing here:', predicted_Img.src)
//            predictionOverlay = new google.maps.GroundOverlay(
//                predicted_Img.src, predictionBounds
//            );
//            predictionOverlay.setMap(map);
//        })
//predicted_Img.src = 'https://storage.cloud.google.com/sketcher-app-test-engine.appspot.com/satelite_screenshots/' + uid + '_test_out_' + count + '.png'
//document.getElementById('staticGrabSpot').appendChild(predicted_Img);
//});


//const btnLogin = document.getElementById('btnLogin');
//const btnLogout = document.getElementById('btnLogout');
//const btnUpload = document.getElementById('fileButton');
//const uploader = document.getElementById('uploader');
//const showPrediction = document.getElementById('showPrediction')
//const toggleDOM = document.getElementById('toggleDOM')

/*
// Click login event listener
btnLogin.addEventListener('click', e => {
firebase.auth().signInAnonymously();
});

// Click logout event listener
btnLogout.addEventListener('click', e => {
firebase.auth().currentUser.delete();
});

firebase.auth().onAuthStateChanged(firebaseUser => {
  if(firebaseUser) {
    var uid = firebaseUser.uid;
    console.log(uid);
    btnLogout.classList.remove('hide');
    //btnUpload.classList.remove('hide');
    //uploader.classList.remove('hide');
  } else {
    btnLogout.classList.add('hide');
    //btnUpload.classList.add('hide');
    //uploader.classList.add('hide');
  }
});

//Code for manually uploading
const fileButton = document.getElementById('fileButton');
// Listen for file selection
fileButton.addEventListener('change', function(e) {
  // Get file
  // Create a storage ref
  var file = e.target.files[0];
  var uid = firebase.auth().currentUser.uid;
  var storageRef = firebase.storage().ref('satelite_screenshots/' + uid);
  //var storageRef = firebase.storage().ref('satelite_screenshots/' + uid + get_url_extension(file.name));
  // Upload file
  var task = storageRef.put(file);
  // Update progress bar
  task.on('state_changed',
    function progress(snapshot){
      var percentage = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
      uploader.value = percentage;
    },
    function error(err) {
    },
    function complete() {
      storageRef.getDownloadURL().then(function(url) {
          var test = url;
          document.querySelector('img').src = test;
      }).catch(function(error) {
      });
    }
  );
});*/




