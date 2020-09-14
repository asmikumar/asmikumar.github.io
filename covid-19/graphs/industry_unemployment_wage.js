// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("industry_unemployment_wage");
var industry_unemployment_wage = new Chart(ctx, {
  type: 'scatter',
  data: {
            datasets: [
                {
                label: 'Scatter Dataset',
                pointRadius: 5,
                pointHoverRadius: 5,
                pointBackgroundColor: ['green', 'red', 'green', 'black', 'red', 'black', 'green', 'red'],
                data: [
                    { x: 201.8, y: 29.35 }, 
                    { x: 354.2, y: 25.28 }, 
                    { x: 237.5, y: 29.01 }, 
                    { x: 323.1, y: 35.44 }, 
                    { x: 589.5, y: 14.55 },
                    { x: 238.5, y: 22.61 }, 
                    { x: 122.7, y: 29.32 }, 
                    { x: 431.4, y: 18.06 }
                ]
                }
            ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          legend: {
            display: false
          },
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom',
                    ticks: {
                    beginAtZero: true
                    }
                }],
                yAxes: [{
                    ticks: {
                    beginAtZero: true
                    }
                }]

            }
        }
    });