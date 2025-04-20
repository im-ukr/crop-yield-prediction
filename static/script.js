document.addEventListener("DOMContentLoaded", function () {
    // Clear All functionality
    document.getElementById("clear-all").addEventListener("click", function () {
        document.querySelectorAll("input").forEach(input => input.value = "");
        document.getElementById("prediction-result").style.display = "none";

        // Clear download button
        const downloadBtn = document.getElementById("download-strategy");
        downloadBtn.style.display = "none";
        document.getElementById("crop-name").textContent = "";
    });

    // Autocomplete functionality with debouncing
    const autocompleteFields = document.querySelectorAll('.autocomplete');
    const debounce = (func, delay) => {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), delay);
        };
    };

    autocompleteFields.forEach(input => {
        const datalist = document.getElementById(input.id + "-list");

        input.addEventListener('input', debounce(function (e) {
            const field = input.dataset.url.split('/').pop();
            const query = e.target.value.trim();

            if (query.length < 2) {
                datalist.innerHTML = '';
                return;
            }

            fetch(`/autocomplete/${field}?query=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(suggestions => {
                    datalist.innerHTML = suggestions
                        .map(suggestion => `<option value="${suggestion}">`)
                        .join('');
                })
                .catch(error => console.error('Autocomplete error:', error));
        }, 300));
    });

    // Form submission handler
    document.getElementById("prediction-form").addEventListener("submit", function (e) {
        e.preventDefault();
        const resultDiv = document.getElementById("prediction-result");
        resultDiv.style.display = "none";

        const formData = {
            State_Name: document.getElementById("State_Name").value,
            District_Name: document.getElementById("District_Name").value,
            Season: document.getElementById("Season").value,
            Crop: document.getElementById("Crop").value,
            Area: document.getElementById("Area").value,
            Crop_Year: document.getElementById("Crop_Year").value
        };

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        })
            .then(response => response.json())
            .then(data => {
                resultDiv.className = "alert alert-info";
                resultDiv.innerHTML = "";

                if (data.error) {
                    resultDiv.className = "alert alert-danger";
                    resultDiv.innerHTML = data.error;
                } else {
                    let resultText = [];
                    if (data.yield) resultText.push(`🌾 Predicted ${formData.Crop} Yield: ${data.yield}`);
                    if (data.crop) resultText.push(`🌱 Recommended Crop: ${data.crop}`);

                    if (resultText.length > 0) {
                        resultDiv.className = "alert alert-success";
                        resultDiv.innerHTML = resultText.join("<br>");
                    } else {
                        resultDiv.className = "alert alert-warning";
                        resultDiv.innerHTML = "No predictions available for the given inputs";
                    }
                }

                resultDiv.style.display = "block";
                // Add this inside the .then(data => { ... }) block after handling the prediction result
                if (data.crop) {
                    const downloadBtn = document.getElementById('download-strategy');
                    const cropNameSpan = document.getElementById('crop-name');

                    // Update button text
                    cropNameSpan.textContent = data.crop;

                    // Show button
                    downloadBtn.style.display = 'inline-block';

                    // Remove existing click handlers
                    downloadBtn.replaceWith(downloadBtn.cloneNode(true));

                    // Add new click handler
                    document.getElementById('download-strategy').addEventListener('click', function () {
                        fetch(`/download-strategy?crop=${encodeURIComponent(data.crop)}`)
                            .then(response => {
                                if (!response.ok) throw new Error('File not found');
                                return response.blob();
                            })
                            .then(blob => {
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `${data.crop}_Harvest_Guide.pdf`;
                                document.body.appendChild(a);
                                a.click();
                                window.URL.revokeObjectURL(url);
                                document.body.removeChild(a);
                            })
                            .catch(error => {
                                alert('Harvest strategy not available for this crop');
                            });
                    });
                } else {
                    document.getElementById('download-strategy').style.display = 'none';
                }
            })
            .catch(error => {
                resultDiv.className = "alert alert-danger";
                resultDiv.innerHTML = "Network Error: Please try again later";
                resultDiv.style.display = "block";
            });
    });
});