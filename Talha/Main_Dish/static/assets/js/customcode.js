$(document).ready(function(){
  var socket = new WebSocket('ws://' + window.location.host + '/ws/main_dish/');

  socket.onopen = function() {
      console.log('connected second page');
  };

  socket.onclose = function() {
      console.log('connection closed');
      $('#loader').hide();
      $(".whisper_load").show();
  };


  socket.onmessage = function(event) {
            var data = JSON.parse(event.data);
            console.log('datatype', data);
            

          
            if (data.status == 'SUCCESS') {

            $('#ingredient').show();
            $('#ingredient').empty();
            $(".whisper_load").show();
            
            $('#loader').hide();
            // Get the response text and split it into an array of ingredients
            var ingredients = data.result['response_text'].split(/\n|\*/);
            
            // Iterate over the ingredients and add them to the list
            ingredients.forEach(function(ingredient) {
                if (ingredient.trim() !== '') {
                    $('#ingredient').append('<li>' + ingredient.trim() + '</li>');
                }
            });
        }else{
          $('#ingredient').hide();
          $('#loader').show();
        }
      };



  $('#upload_video_file_whisper').on('click', function(){

    const fileInput = document.getElementById('formFile_video_whisper');
      const file = fileInput.files[0];

      if (!file) {
        alert('No file selected. Please choose a file to upload.');
        return; // Exit the function if no file is selected
    }



      $("#loader").show();
      $('#ingredient').empty();
      $(".whisper_load").hide();
      if (file) {
          const reader = new FileReader();
          reader.onload = function(event) {
              const fileContent = event.target.result;
              const data = {
                  type: 'whisper_video',
                  file: {
                      name: file.name,
                      content: Array.from(new Uint8Array(fileContent))
                  }
              };
              socket.send(JSON.stringify(data));
          };
          reader.readAsArrayBuffer(file);
      }


  })



})