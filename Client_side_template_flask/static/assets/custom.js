
$(document).ready(function(){

    $("#uploadForm").on("submit", function(event) {

            event.preventDefault();  // Prevent default form submission
            
            $("#statusMessage").text('');
            $("#status").text('');
            var formData = new FormData();
            var fileInput = $("#allFile")[0].files[0];

            if (!fileInput) {
                $("#statusMessage").text("Please select a file.");
                return;
            }

            formData.append("bookdata", fileInput);

            $.ajax({
                url: "/upload",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {

                    $("#statusMessage").text(response.message);
                },
                error: function(xhr) {
                    $("#statusMessage").text("Error: " + xhr.responseJSON.error);
                }
            });
        });



        ///// rag jquery ///

        $("#queryForm").on("submit", function(event) {
        $("#loader_for_rag_response").show();
        $("#rag_submit").prop('disabled', true);
        event.preventDefault();  // Prevent default form submission
        $("#statusMessage").text('');
        $("#status").text('');
        
        var userQuery = $("#user_query").val(); // Get text input value

        if (!userQuery) {
            $("#statusMessage").text("Please enter a query.");
            return;
        }

        $.ajax({
            url: "/Model_output",
            type: "POST",
            contentType: "application/json",  // ✅ Send as JSON
            data: JSON.stringify({ "rag_query": userQuery }), // ✅ Convert to JSON format
            success: function(response) {
                $("#rag_submit").prop('disabled', false);
                $("#loader_for_rag_response").hide();
                console.log("response", response.result)
                $("#statusMessage").text(response.result);
            },
            error: function(xhr) {
                $("#statusMessage").text("Error: " + (xhr.responseJSON?.error || "Unknown error"));
            }
        });
    });


        //// rag jquery close ///


        /// video summarization open ///


        $("#uploadvideo").on("submit", function(event) {
           $("#loader").show();
           $("#Summarization_submit").prop('disabled', true);
            event.preventDefault();  // Prevent default form submission
            $("#statusMessage_summarization").text('');
            $("#status").text('');
            var formData = new FormData();
            var fileInput = $("#video_file")[0].files[0];

            if (!fileInput) {
                $("#statusMessage_summarization").text("Please select a file.");
                return;
            }

            formData.append("video_file", fileInput);

            $.ajax({
                url: "/video_summarization",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                    $("#Summarization_submit").prop('disabled', false);
                    $("#loader").hide();
                    console.log("response", response);
                    $("#statusMessage_summarization").text(JSON.stringify(response.result, null, 2));

                },
                error: function(xhr) {
                    $("#statusMessage_summarization").text("Error: " + xhr.responseJSON.error);
                }
            });
        });


        /// video summarization close ///

        // translation jquery open //

        $("#textForm").on("submit", function(event) {
        $("#loader_for_translation").show();
        $("#translation_submit").prop('disabled', true);
        event.preventDefault();  // Prevent default form submission
        $("#statusMessage").text('');
        $("#status").text('');
        var src_language = $("#src_language").val();
        var tgt_language = $("#tgt_language").val();
        console.log("data", src_language, tgt_language);
        var userQuery = $("#text").val(); // Get text input value

        var requestData = { 
            "type": "text",
            "src_language": src_language, 
            "tgt_language": tgt_language, 
            "text": userQuery 
        };
        console.log("Sending Data:", requestData);

        if (!userQuery) {
            $("#statusMessage").text("Please enter a query.");
            return;
        }

        $.ajax({
            url: "/translation",
            type: "POST",
            contentType: "application/json",  // ✅ Send as JSON
            data: JSON.stringify(requestData),
            success: function(response) {
                $("#loader_for_translation").hide();
                $("#translation_submit").prop('disabled', false);
                console.log("response", response.result);
                $("#floatingTextarea2").text(response.result);
            },
            error: function(xhr) {
                $("#statusMessage").text("Error: " + (xhr.responseJSON?.error || "Unknown error"));
            }
        });
    });






    $("#translation_form").on("submit", function(event) {
        $("#loader_for_file_translation").show();
        $("#translation_file_submit").prop('disabled',true);
        // alert('file')
        event.preventDefault();  // Prevent default form submission
        $("#statusMessage").text('');
        $("#status").text('');
        var formData = new FormData();
        var fileInput = $("#translation_file")[0].files[0];
        var src_language = $("#src_language_file").val();
        var tgt_language = $("#tgt_language_file").val();

        console.log("srsss",src_language);

        if (!fileInput) {
            $("#statusMessage").text("Please select a file.");
            return;
        }

        formData.append("translation_file", fileInput);

        formData.append("type", "file");
        
        formData.append("src_lang_file", src_language);

        formData.append("tgt_lang_file", tgt_language);

        $.ajax({
            url: "/translation",
            type: "POST",
            data: formData,
            processData: false, // Prevent jQuery from processing data
            contentType: false, // Let the browser set the content type
            success: function(response) {
                $("#translation_file_submit").prop('disabled',false);
                $("#loader_for_file_translation").hide();
                console.log("response", response)
            },
            error: function(xhr) {
                $("#statusMessage").text("Error: " + xhr.responseJSON.error);
            }
        });
    });



        // translation jquery close //



        //transcription jquery open //

        $("#uploadaudio").on("submit", function(event) {
            $("#loader_for_transcription").show();
            $("#transcription_submit").prop('disabled', true);
            event.preventDefault();  // Prevent default form submission
            $("#statusMessage").text('');
            $("#status").text('');
            var formData = new FormData();
            var fileInput = $("#audio_file")[0].files[0];
            var translanguage = $("#trans_language").val();
            // alert(`trans, ${translanguage}`);

            if (!fileInput) {
                $("#statusMessage").text("Please select a file.");
                return;
            }

            formData.append("audio_file", fileInput);
            formData.append('trans_language', translanguage);

            $.ajax({
                url: "/transcription",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                    $("#transcription_submit").prop('disabled', false);
                    $("#loader_for_transcription").hide()
                    console.log("response", response)
                    $("#transcription_results").text(response.result);
                },
                error: function(xhr) {
                    $("#statusMessage").text("Error: " + xhr.responseJSON.error);
                }
            });
        });


        // transcription jquery close //


        // fake video jquery open //


        $(document).on("submit", "#fakevideo_upload", function (event) {
            $("#loader_for_fakevideo").show();
            $("#fake_submit").prop('disabled', true);
            console.log("Form submitted via event delegation");
            event.preventDefault();
            // Rest of your AJAX logic here...


            $("#statusMessage222").text('');
            $("#status").text('');
            var formData = new FormData();
            var fileInput = $("#fake_video_file")[0].files[0];
            console.log("fileinput", fileInput);
            if (!fileInput) {
                $("#statusMessage222").text("Please select a file.");
                return;
            }

            formData.append("fake_video_file", fileInput);
            console.log("Sending request...");

            $.ajax({
                url: "/fake_video",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                    $("#fake_submit").prop('disabled', false);
                    $("#loader_for_fakevideo").hide();
                    console.log("response", response);
                
                    if (response && response.result && response.result.length > 0) {
                        $("#statusMessage222").text(JSON.stringify(response.result, null, 2));
                    } else {
                        $("#statusMessage222").text("No result found.");
                    }
                },
                
                error: function(xhr) {
                    $("#statusMessage222").text("Error: " + xhr.responseJSON.error);
                }
            });


        });





        // fake video jquery close //




        

        // video converter jquery open //



        document.getElementById('video_converter_file').addEventListener('change', function(e) {
            var fileInput = e.target;  // Get the file input element itself
            
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0]; // Get the first file
        
                // Get the file name and extension
                var fileName = file.name;
                var fileExtension = fileName.split('.').pop().toLowerCase();
                console.log("File Extension: ", fileExtension);
        
                // Optionally: Validate the file extension
                const validation = validateVideoFile(file);
                if (!validation.valid) {
                    alert(validation.error);
                    e.target.value = ''; // Reset the input field
                    return; // Stop the process if the validation fails
                }
        
                // Display video info
                displayVideoInfo(fileInput);  // Pass fileInput, not file
        
                // Update the dropdown options
                updateFormatDropdown(fileExtension);
            }
        });
        



        $(document).on("submit", "#video_converter", function (event) {
            $("#loader_for_fakevideo").show();
            $("#converter_submit").prop('disabled', true);
            $('.video-info').hide();
            console.log("Form submitted via event delegation");
            event.preventDefault();
        
            // Show the progress bar
            $("#progressBarContainer").show();
            updateProgressBar(0);  // Start with 0% progress
        
            var formData = new FormData();
            var fileInput = $("#video_converter_file")[0].files[0];



            var video_format = $("#video_format").val();
            console.log("fileinput", fileInput);
        
            if (!fileInput) {
                $("#statusMessage_video_converter").text("Please select a file.");
                return;
            }
        
            formData.append("video_converter_file", fileInput);
            formData.append("video_format", video_format);
            console.log("Sending request...");
        
            // Simulate the progress updates right after clicking the button
            updateProgressBar(10);  // Start progress immediately at 10%
        
            // Simulate a step-by-step progress (adjust these intervals as needed)
            setTimeout(function() {
                updateProgressBar(30);  // 30% after 2 seconds
                setTimeout(function() {
                    updateProgressBar(60);  // 60% after 2 more seconds
                    setTimeout(function() {
                        updateProgressBar(90);  // 90% after another 2 seconds
                    }, 5000);
                }, 4000);
            }, 5000);
        
            $.ajax({
                url: "/Video_Converter",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                    $("#converter_submit").prop('disabled', false);
                    $("#loader_for_fakevideo").hide();
                    console.log("response", response);
        
                    // Handle the final 100% progress and show the download link
                    updateProgressBar(100);  // Set progress to 100%
        
                    $("#progressBarContainer").hide();


                    if (response && response.download_url) {
                        var downloadLink = $("<a></a>").attr({
                            href: response.download_url,
                            download: response.filename  // Use the filename from the response for the download
                        }).addClass("btn btn-success").text("Download Converted Video");

                        $("#statusMessage_video_converter").html("Video conversion completed. ").append(downloadLink);
                    } else {
                        $("#statusMessage_video_converter").text("No result found.");
                    }
                },
                error: function(xhr) {
                    $("#statusMessage_video_converter").text("Error: " + xhr.responseJSON.error);
                    updateProgressBar(0);  // Reset progress bar on error
                }
            });
        });
        
        // Function to update progress bar
        function updateProgressBar(percentage) {
            $("#progressBar").css('width', percentage + '%').attr('aria-valuenow', percentage).text(percentage + '%');
        }
        


        function updateProgressBar_compress(percentage) {
            $("#progressBar_compresser").css('width', percentage + '%').attr('aria-valuenow', percentage).text(percentage + '%');
        }
        





        function validateVideoFile(file) {
            const allowedFormats = {
                extensions: ['dav', 'mp4', 'avi', 'mov', 'mkv'],
                mimeTypes: ['video/dav', 'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/avi']
            };
    
            const extension = file.name.split('.').pop().toLowerCase();
            
            if (!allowedFormats.extensions.includes(extension)) {
                return {
                    valid: false,
                    error: `Invalid file format. Please upload a video in one of these formats: ${allowedFormats.extensions.join(', ').toUpperCase()}`
                };
            }
    
            // For DAV files, we accept them directly since they might not have a standard MIME type
            if (extension === 'dav') {
                return { valid: true };
            }
    
            // Check MIME type for non-DAV files
            if (!allowedFormats.mimeTypes.includes(file.type)) {
                return {
                    valid: false,
                    error: 'Invalid video file. Please upload a valid video file.'
                };
            }
    
            return { valid: true };
        }

        

        function updateFormatDropdown(currentExtension) {
            const formatSelect = document.getElementById('video_format');
            Array.from(formatSelect.options).forEach(option => {
                if (option.value.toLowerCase() === currentExtension) {
                    option.hidden = true; // Hide the current format option
                } else {
                    option.hidden = false; // Show other formats
                }
            });
        
            // If the selected option is hidden, select a different one
            if (formatSelect.options[formatSelect.selectedIndex].hidden) {
                formatSelect.selectedIndex = 0; // Select the first visible option
            }
        }



        function displayVideoInfo(fileInput) {
            const file = fileInput.files[0];  // Get the first file selected
            if (file) {
                const videoInfoDiv = fileInput.parentElement.querySelector('.video-info') || 
                    document.createElement('div');
                videoInfoDiv.className = 'video-info mt-2';
                videoInfoDiv.innerHTML = `
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title text-primary">
                                <i class="fas fa-info-circle me-2"></i>Video Information
                            </h6>
                            <div class="card-text">
                                <div class="mb-1"><strong>Name:</strong> ${file.name}</div>
                                <div class="mb-1"><strong>Size:</strong> ${(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                                <div><strong>Type:</strong> ${file.type || 'video/dav'}</div>
                            </div>
                        </div>
                    </div>
                `;
                if (!fileInput.parentElement.querySelector('.video-info')) {
                    fileInput.parentElement.appendChild(videoInfoDiv);
                }
            }
        }
        

    


        
        /*$(document).on("submit", "#video_converter", function (event) {
            $("#loader_for_fakevideo").show();
            $("#converter_submit").prop('disabled', true);
            console.log("Form submitted via event delegation");
            event.preventDefault();
          

            $("#statusMessage_video_converter").text('');
            $("#status").text('');
            var formData = new FormData();
            var fileInput = $("#video_converter_file")[0].files[0];
            var video_format = $("#video_format").val();
            console.log("fileinput", fileInput);
            if (!fileInput) {
                $("#statusMessage_video_converter").text("Please select a file.");
                return;
            }

            formData.append("video_converter_file", fileInput);
            formData.append("video_format", video_format);
            console.log("Sending request...");

            $.ajax({
                url: "/Video_Converter",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                    $("#converter_submit").prop('disabled', false);
                    $("#loader_for_fakevideo").hide();
                    console.log("response", response);
                
                    if (response) {


                        var downloadLink = $("<a></a>").attr({
                            href: response.download_url, 
                            download: response.filename  // Use the filename from the response for the download
                        }).text("Download Converted Video");

                        $("#statusMessage_video_converter").html("Video conversion completed. ").append(downloadLink);



                    } else {
                        $("#statusMessage_video_converter").text("No result found.");
                    }
                },
                
                error: function(xhr) {
                    $("#statusMessage222").text("Error: " + xhr.responseJSON.error);
                }
            });


        });*/
        

        // video converter jquery close //




        // video compresser jquery code open //


        document.getElementById('video_compresser_file').addEventListener('change', function(e) {
            var fileInput = e.target;  // Get the file input element itself
            
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0]; // Get the first file
        
                // Get the file name and extension
                var fileName = file.name;
                var fileExtension = fileName.split('.').pop().toLowerCase();
                console.log("File Extension: ", fileExtension);
        
                // Optionally: Validate the file extension
                const validation = validateVideoFile(file);
                if (!validation.valid) {
                    alert(validation.error);
                    e.target.value = ''; // Reset the input field
                    return; // Stop the process if the validation fails
                }
        

        
                // Update the dropdown options

            }
        });
        






        $(document).on("submit", "#video_compresser", function (event) {

            $("#compresser_submit").prop('disabled', true);
            $('.video-info').hide();
            console.log("Form submitted via event delegation");
            event.preventDefault();
        
            $("#progressBarContainer_compresser").show();
            updateProgressBar_compress(0);  // Start with 0% progress


        
            var formData = new FormData();
            var fileInput = $("#video_compresser_file")[0].files[0];
            const compressionRate = document.getElementById('compressionRate').value;
            
            console.log("compressionRate",compressionRate);
            

            console.log("fileinput", fileInput);
        
            if (!fileInput) {
                $("#statusMessage_video_compresser").text("Please select a file.");
                return;
            }
        
            formData.append("video_compresser_file", fileInput);
            formData.append("compress_rate", compressionRate);
            console.log("Sending request...");
        
            // Simulate the progress updates right after clicking the button


            updateProgressBar_compress(10);  // Start progress immediately at 10%
        
            // Simulate a step-by-step progress (adjust these intervals as needed)
            setTimeout(function() {
                updateProgressBar_compress(30);  // 30% after 2 seconds
                setTimeout(function() {
                    updateProgressBar_compress(60);  // 60% after 2 more seconds
                    setTimeout(function() {
                        updateProgressBar_compress(90);  // 90% after another 2 seconds
                    }, 5000);
                }, 4000);
            }, 5000);
        



            $.ajax({
                url: "/Video_Compresser",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                    $("#compresser_submit").prop('disabled', false);
                    console.log("response", response);
                    updateProgressBar_compress(100); 

                    $("#progressBarContainer_compresser").hide();

                    if (response && response.download_url) {
                        var downloadLink = $("<a></a>").attr({
                            href: response.download_url,
                            download: response.filename  // Use the filename from the response for the download
                        }).addClass("btn btn-warning").text("Download Compress Video");

                        $("#statusMessage_video_compresser").html("Video Compression completed. ").append(downloadLink);
                    } else {
                        $("#statusMessage_video_compresser").text("No result found.");
                    }




                },
                error: function(xhr) {
                    $("#statusMessage_video_compresser").text("Error: " + xhr.responseJSON.error);
                    updateProgressBar_compress(0);  // Reset progress bar on error

                
                }
            });
        });
        


        // video compresser jquery code close //



        // object detection jquery code //





        $(document).on("submit", "#detection_file", function (event) {

            $("#loader_for_object_detection").show();
           
            $("#detection_submit").prop('disabled', true);
            console.log("Form submitted via event delegation");
            $("#detectionTable").empty();
            event.preventDefault();


            var selectedValues = [];
            $('input[type="checkbox"]:checked').each(function () {
                var value = $(this).val();
                var name = $(this).next('label').text();  // Get the label text associated with the checkbox
                selectedValues.push({ value: value, name: name });  // Sto


            });
    
            console.log("Checkbox values on form submit:", selectedValues);


            $("#object_detection_text").text('');
            $("#status").text('');
            var formData = new FormData();
            var fileInput = $("#object_detection_file")[0].files[0];
            console.log("fileinput", fileInput);
            if (!fileInput) {
                $("#object_detection_text").text("Please select a file.");
                return;
            }

            formData.append("object_detection_file", fileInput);
            formData.append("selected_checkboxes", JSON.stringify(selectedValues));
            console.log("Sending request...");

            $.ajax({
                url: "/object_detection",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                    $("#detection_submit").prop('disabled', false);
                    $("#table_results").show();
                    $("#loader_for_object_detection").hide();
                    let message = response.result.message;  // Get message
                    let detections = response.result.results;  // Get detections array
                
                
                    // Show message in an alert or div (optional)
                    $("#detectionMessage").text(message);  
                
                    // Clear the table
                    $("#detectionTable").empty();
                
                    if (!Array.isArray(detections) || detections.length === 0) {
                        $("#detectionTable").append("<tr><td colspan='6' class='text-center'>No detections found.</td></tr>");
                    } else {
                       
                        detections.forEach(function(item) {
                            let row = `
                                <tr>
                                    <td>${item[0]}</td>  <!-- Date -->
                                    <td>${item[1]}</td>  <!-- Time -->
                                    <td>${item[2]}</td>  <!-- Class -->
                                    <td>${item[3]}</td>  <!-- Track ID -->
                                    <td>${(item[4] * 100).toFixed(2)}%</td>  <!-- Confidence -->
                                </tr>
                            `;
                            $("#detectionTable").append(row);
                        });


                    }
                },
                
                error: function(xhr) {
                    $("#object_detection_text").text("Error: " + xhr.responseJSON.error);
                }
            });


        });




        // object detection jquery code close //



        // Image enhance jquery code open //



        $(document).on("submit", "#image_enhance_form", function (event) {
            $("#image_enhance_submit").prop('disabled', true);
            console.log("Form submitted via event delegation");
            event.preventDefault();
            // Rest of your AJAX logic here...



            var formData = new FormData();
            var fileInput = $("#image_enhance_file")[0].files[0];
            console.log("fileinput", fileInput);
            if (!fileInput) {
                $("#image_enhance_222").text("Please select a file.");
                return;
            }

            formData.append("object_detection_file", fileInput);
            formData.append('type', 'image_file')
            console.log("Sending request...");

            $.ajax({
                url: "/object_enhance",
                type: "POST",
                data: formData,
                processData: false, // Prevent jQuery from processing data
                contentType: false, // Let the browser set the content type
                success: function(response) {
                   
                    console.log("response", response);

                    let imgElement = document.getElementById("enhancedImage");
                    imgElement.src = response.file_url;  // Set the image source

                },
                
                error: function(xhr) {
                    $("#image_enhance_222").text("Error: " + xhr.responseJSON.error);
                }
            });


        });





        // Image enhance jquery code close //





            // Video enhance jquery code open //



            $(document).on("submit", "#video_enhance_form", function (event) {
                $("#video_enhance_submit").prop('disabled', true);
                console.log("Form submitted via event delegation");
                event.preventDefault();
                // Rest of your AJAX logic here...
    
    
    
                var formData = new FormData();
                var fileInput = $("#video_enhance_file")[0].files[0];
                console.log("fileinput", fileInput);
                if (!fileInput) {
                    $("#video_enhance_222").text("Please select a file.");
                    return;
                }
    
                formData.append("object_detection_file", fileInput);
                formData.append('type', 'video_file')
                console.log("Sending request...");
    
                $.ajax({
                    url: "/object_enhance",
                    type: "POST",
                    data: formData,
                    processData: false, // Prevent jQuery from processing data
                    contentType: false, // Let the browser set the content type
                    success: function(response) {
                       
                        console.log("response", response);
                
                    },
                    
                    error: function(xhr) {
                        $("#video_enhance_222").text("Error: " + xhr.responseJSON.error);
                    }
                });
    
    
            });
    
    
    
    
    
            // Video enhance jquery code close //


            // crowd counting image jquery code open //



            $("#crowd_counting_image").on("submit", function(event) {
               
                $("#crowd_image_submit").prop('disabled', true);
                event.preventDefault();  // Prevent default form submission
              
                var formData = new FormData();
                var fileInput = $("#crowd_image_file")[0].files[0];
                console.log("fileinput", fileInput);
                if (!fileInput) {
                    $("#video_enhance_222").text("Please select a file.");
                    return;
                }
    
                formData.append("crowd_image_file", fileInput);
                formData.append('type', 'image_file')
                console.log("Sending request...");

                $.ajax({
                    url: "/Crowd_Counting",
                    type: "POST",
                    data: formData,
                    processData: false, // Prevent jQuery from processing data
                    contentType: false, // Let the browser set the content type
                    success: function(response) {
                       
                        console.log("response", response);
                
                    },
                    
                    error: function(xhr) {
                        $("#video_enhance_222").text("Error: " + xhr.responseJSON.error);
                    }
                });


            });





            // crowd counting image jquery code close //








            
            // crowd counting video jquery code open //



            $("#crowd_counting_video").on("submit", function(event) {
               
                $("#crowd_video_submit").prop('disabled', true);
                event.preventDefault();  // Prevent default form submission
              
                var formData = new FormData();
                var fileInput = $("#crowd_video_file")[0].files[0];
                console.log("fileinput", fileInput);
                if (!fileInput) {
                    $("#video_enhance_222").text("Please select a file.");
                    return;
                }
    
                formData.append("crowd_image_file", fileInput);
                formData.append('type', 'video_file')
                console.log("Sending request...");

                $.ajax({
                    url: "/Crowd_Counting",
                    type: "POST",
                    data: formData,
                    processData: false, // Prevent jQuery from processing data
                    contentType: false, // Let the browser set the content type
                    success: function(response) {
                       
                        console.log("response", response);
                
                    },
                    
                    error: function(xhr) {
                        $("#video_enhance_222").text("Error: " + xhr.responseJSON.error);
                    }
                });


            });





            // crowd counting video jquery code close //



})