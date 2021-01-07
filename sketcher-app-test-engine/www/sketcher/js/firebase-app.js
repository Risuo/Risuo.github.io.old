
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
const btnUpload = document.getElementById('fileButton');
const uploader = document.getElementById('uploader');

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
        btnUpload.classList.remove('hide');
        uploader.classList.remove('hide');
      } else {
        btnLogout.classList.add('hide');
        btnUpload.classList.add('hide');
        uploader.classList.add('hide');
      }
  });



function get_url_extension( url ) {
    return url.split(/[#?]/)[0].split('.').pop().trim();
}

// Code for uploading

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

});