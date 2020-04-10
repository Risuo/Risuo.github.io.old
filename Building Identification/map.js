var map2;
var chicago = {lat: 41.85, lng: -87.65};

/**
 * The CenterControl adds a control to the map that recenters the map on
 * Chicago.
 * This constructor takes the control DIV as an argument.
 * @constructor
 */
function CenterControl2(controlDiv2, map2) {

  // Set CSS for the control border.
  var controlUI = document.createElement('div');
  controlUI.style.backgroundColor = '#33FFBB';
  controlUI.style.border = '2px solid #000000';
  controlUI.style.borderRadius = '30px';
  controlUI.style.boxShadow = '0 2px 6px rgba(0,0,0,.3)';
  controlUI.style.cursor = 'pointer';
  controlUI.style.marginBottom = '22px';
  controlUI.style.marginTop = '5px'
  controlUI.style.textAlign = 'center';
  controlUI.title = 'Click to perform Building Inference!';
  controlDiv2.appendChild(controlUI);

  // Set CSS for the control interior.
  var controlText = document.createElement('div');
  controlText.style.color = 'rgb(25,25,25)';
  controlText.style.fontFamily = 'Roboto,Arial,sans-serif';
  controlText.style.fontSize = '20px';
  controlText.style.lineHeight = '38px';
  controlText.style.paddingLeft = '5px';
  controlText.style.paddingRight = '5px';
  controlText.innerHTML = 'Grab Satellite Image';
  controlUI.appendChild(controlText);

  // Setup the click event listeners: simply set the map to Chicago.
  controlUI.addEventListener('click', function() {
    map2.setCenter(chicago);
  });
}




function initMap() {
  map2 = new google.maps.Map(document.getElementById('map2'), {
    zoom: 12,
    center: chicago,
    mapTypeId: 'satellite',
    gestureHandling: 'greedy',
    streetViewControl: false,
    scaleControl: true,
    fullscreenControl: false,
    mapTypeControlOptions: {
    mapTypeIds: ['']
    }
  });

  // Create the DIV to hold the control and call the CenterControl()
  // constructor passing in this DIV.
  var centerControlDiv = document.createElement('div');
  var centerControl2 = new CenterControl2(centerControlDiv, map2);

  centerControlDiv.index = 1;
  map2.controls[google.maps.ControlPosition.TOP_CENTER].push(centerControlDiv);

}


