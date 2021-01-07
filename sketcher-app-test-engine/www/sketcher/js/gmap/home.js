var map;


function initialize() {
    //var latlng = new google.maps.LatLng(53.396540795528246,-7.942222549768076);
    //var latlng = new google.maps.LatLng(53.396540795528246,-7.942222549768076);
    //var latlng = new google.maps.LatLng(53.396540795528246,-7.942222549768076);
    var latlng = new google.maps.LatLng(38.8976925,-77.0368151);
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
            map.getCenter() //
            // +
            //" lat: " +
            //(event.latLng.lat()) +
            //", " +
            //"lng: " +
            //(event.latLng.lng());
    });
    // Add a marker to the map for the end-point of the directions.
    /*
     var marker = new google.maps.Marker({
     position: latlng,
     map: map,
     title:"Rodderhof, Oss"
     });
     */
}
$(document).ready(function() {
    var autocompleteLoaded = 0;
    var autocomplete;

    $('#street_address').keyup(function() {
        if (autocompleteLoaded==1 && this.value.length<7)
        {
            autocomplete.unbindAll();
            google.maps.event.clearInstanceListeners(document.getElementById('street_address'));
            $(".pac-container").hide();
            autocompleteLoaded=0;
        }
        if (autocompleteLoaded==0 && this.value.length>=7)
        {
            autocompleteLoaded=1;
            var input = document.getElementById('street_address');
            var options = {
            };
            autocomplete = new google.maps.places.Autocomplete(input, options);

            google.maps.event.addListener(autocomplete, 'place_changed', function() {});
        }
    });

    $("#frm_search_address").submit(function() {

        var street_address = $("#street_address").val();
        if(street_address.length > 0 ){
            // display code address
            showAddress(street_address);
        }

        return false;
    });

});