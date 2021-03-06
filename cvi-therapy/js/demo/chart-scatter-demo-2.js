// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont, Segoe UI, Avenir, sans-serif, Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("scatterSpeed");
var scatterSpeed = new Chart(ctx, {
  type: 'scatter',
  data: {
            datasets: [

                // low = 50 - <100, medium = 100 - <150, high = 150 - 200
                {
                  data: [
                      { x: 50, y: 76 }, 
                      { x: 50, y: 79 }, 
                      { x: 50, y: 75 },
                      { x: 75, y: 74 }, 
                      { x: 80, y: 74 },
                      { x: 50, y: 81 }, 
                      { x: 60, y: 80 },  
                  ],
                  backgroundColor: '#89cfef',
                  borderWidth: 0,
                  label: 'Low',
                },

                {
                  data: [
                      { x: 110, y: 66 }, 
                      { x: 120, y: 67 },
                      { x: 125, y: 61 },
                      { x: 120, y: 65 },
                      { x: 120, y: 64 },
                  ],
                  backgroundColor: '#0492c2',
                  borderWidth: 0,
                  label: 'Medium',
                },

                {
                  data: [
                      { x: 150, y: 32 },
                      { x: 150, y: 37 },
                      { x: 200, y: 30 },
                  ],
                  backgroundColor: '#1338be',
                  borderWidth: 0,
                  label: 'High',
                },  

            ]
        },
        options: {
          legend: {
            display: true,
            onClick: null
          },
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom',
                    ticks: {
                      beginAtZero: false,
                      steps: 17,
                      stepSize: 10,
                      min: 50,
                      max: 200,
                    },
                    scaleLabel: {
                      display: true,
                      labelString: 'Speed (px/sec)'
                    },
                }],
                yAxes: [{
                    type: 'linear',
                    position: 'bottom',
                    ticks: {
                      steps: 11,
                      stepSize: 10,
                      min: 0,
                      max: 100,
                    },
                    scaleLabel: {
                      display: true,
                      labelString: 'Score (%)'
                    }
                }]
            }
        }

    });