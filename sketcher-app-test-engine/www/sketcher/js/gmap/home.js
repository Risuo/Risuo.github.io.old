/*
var map;
var centerUrl;
var mapZoom;


function initialize() {
    //var latlng = new google.maps.LatLng(53.396540795528246,-7.942222549768076);
    //var latlng = new google.maps.LatLng(53.396540795528246,-7.942222549768076);
    //var latlng = new google.maps.LatLng(53.396540795528246,-7.942222549768076);
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
    map.addListener("mousemove", (event) => {
        coordsDiv.textContent =
            "Center: " +
            map.getCenter() +
            //" Bounds: " +
            //map.getBounds() +
            " Zoom: " +
            map.getZoom()//
            // +
            //" lat: " +
            //(event.latLng.lat()) +
            //", " +
            //"lng: " +
            //(event.latLng.lng());
    });
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

    const btnGrabMap = document.getElementById('btnGrabMap');


// Click button to grab static image & display based on variables on dynamic map
    btnGrabMap.addEventListener('click', e => {
        console.log(centerUrl, mapZoom);
        var static_Img = document.createElement('img');
        static_Img.src = 'https://maps.googleapis.com/maps/api/staticmap?center=' + centerUrl + '&zoom=' + mapZoom + '&maptype=satellite' + '&size=640x640' + '&key=AIzaSyCiTASyv4ikDvjz3nRgbGNiUAn-Z4MOLlI&v=3.exp&libraries=places';
        console.log(static_Img.src)
        document.getElementById('staticGrabSpot').appendChild(static_Img);
    });

    // Add a marker to the map for the end-point of the directions.
    /*
     var marker = new google.maps.Marker({
     position: latlng,
     map: map,
     title:"Rodderhof, Oss"
     });
     */
/*
}

*/
