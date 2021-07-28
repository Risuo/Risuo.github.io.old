
var map;
var centerUrl;
var mapZoom;
var static_Img;
var blob;


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
//firebase.analytics();

// Code for anonymous login

// Get elements
const btnLogin = document.getElementById('btnLogin');
const btnLogout = document.getElementById('btnLogout');
//const btnUpload = document.getElementById('fileButton');
//const uploader = document.getElementById('uploader');


// Click login event listener
  btnLogin.addEventListener('click', e => {
    firebase.auth().signInAnonymously();
});

  // Click logout event listener
  btnLogout.addEventListener('click', e => {
    firebase.auth().currentUser.delete();
    //firebase.auth().signOut();
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

function get_url_extension( url ) {
    return url.split(/[#?]/)[0].split('.').pop().trim();
}


/*
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

function initialize() {
    var latlng = new google.maps.LatLng(38.89763405057457, -77.03643959073791);
    // set direction render options
    var rendererOptions = { draggable: true };
    directionsDisplay = new google.maps.DirectionsRenderer(rendererOptions);
    var myOptions = {
        zoom: 19,
        center: latlng,
        mapTypeId: 'satellite',
        gestureHandling: 'greedy',
        streetViewControl: false,
        scaleControl: true,
        fullscreenControl: false,
        mapTypeControlOptions: {
        mapTypeIds: ['']
        }
    };
    // add the map to the map placeholder
    map = new google.maps.Map(document.getElementById("map_canvas"),myOptions);
    directionsDisplay.setMap(map);
    directionsDisplay.setPanel(document.getElementById("directionsPanel"));
    map.setTilt(0);
    const coordsDiv = document.getElementById("coords");
    map.controls[google.maps.ControlPosition.TOP_CENTER].push(coordsDiv);
    //map.addListener("mousemove", (event) => {
    //    coordsDiv.textContent =
            //"Center: " +
            //map.getCenter() +
            //" Bounds: " +
            //map.getBounds() +
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
        console.log(centerUrl, mapZoom)
    });

    map.addListener("mouseout", (event) => {
        centerUrl = map.getCenter().toUrlValue();
        mapZoom = map.getZoom();
        console.log(centerUrl, mapZoom)
    });

    const btnMLBegin = document.getElementById('btnMLBegin');


// Click button to grab static image & display based on variables on dynamic map & upload to Firebase storage & retrieve URL for static map

    btnMLBegin.addEventListener('click', e => {
        console.log(centerUrl, mapZoom);
        static_Img = document.createElement('img');
        static_Img.src = 'https://maps.googleapis.com/maps/api/staticmap?center=' + centerUrl + '&zoom=' + mapZoom + '&maptype=satellite' + '&size=640x640' + '&format=png32' + '&key=AIzaSyCiTASyv4ikDvjz3nRgbGNiUAn-Z4MOLlI&v=3.exp&libraries=places';
        document.getElementById('staticGrabSpot').appendChild(static_Img);
        var static_Url = static_Img.src;
        console.log('static_Img URL:', static_Url)
        var uid = firebase.auth().currentUser.uid;
        var filename = firebase.storage().ref('satelite_screenshots/' + uid);

        fetch(static_Url).then(res => {
            return res.blob();
        }).then(blob => {
            filename.put(blob).then(function(snapshot) {
                return snapshot.ref.getDownloadURL()
            }).then(url => {
                console.log("Firebase storage image uploaded available at: ", url);
            })
        }).catch(error => {
            console.error(error);
        });
    });
}


