// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Area Chart Example
var ctx = document.getElementById("unemployment_quarterly");
var unemployment_quarterly = new Chart(ctx, {
  type: 'line',
  data: {
    labels: ["Q2/2017", "Q3/2017", "Q4/2017",
            "Q1/2018", "Q2/2018", "Q3/2018", "Q4/2018",
            "Q1/2019", "Q2/2019", "Q3/2019", "Q4/2019",
            "Q1/2020", "Q2/2020"],
    datasets: [
    {
      label: "White",
      lineTension: false,
      fill: false,
      backgroundColor: "rgba(0,153,255,0.2)",
      borderColor: "rgba(0,153,255,1)",
      pointRadius: 0,
      pointBackgroundColor: "rgba(0,153,255,1)",
      pointBorderColor: "rgba(255,255,255,0.8)",
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(0,153,255,1)",
      pointHitRadius: 50,
      pointBorderWidth: 2,
      data: [3.8,3.7,3.6,3.6,3.5,3.3,3.4,3.4,3.2,3.3,3.2,3.4,12.2],
    },
    {
      label: "Black",
      lineTension: false,
      fill: false,
      backgroundColor: "rgba(102,102,204,0.2)",
      borderColor: "rgba(102,102,204,1)",
      pointRadius: 0,
      pointBackgroundColor: "rgba(102,102,204,1)",
      pointBorderColor: "rgba(255,255,255,0.8)",
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(102,102,204,1)",
      pointHitRadius: 50,
      pointBorderWidth: 2,
      data: [7.6,7.4,7.2,7.0,6.3,6.4,6.3,6.8,6.2,5.6,5.6,6.2,16.3],
    },
    {
      label: "Hispanic",
      lineTension: false,
      fill: false,
      backgroundColor: "rgba(255,204,0,0.2)",
      borderColor: "rgba(255,204,0,1)",
      pointRadius: 0,
      pointBackgroundColor: "rgba(255,204,0,1)",
      pointBorderColor: "rgba(255,255,255,0.8)",
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(255,204,0,1)",
      pointHitRadius: 50,
      pointBorderWidth: 2,
      data: [5.0,5.1,4.9,5.0,4.8,4.6,4.4,4.6,4.2,4.2,4.1,4.9,17.0],
    },

    {
      label: "Men",
      lineTension: false,
      fill: false,
      backgroundColor: "rgba(102,204,0,0.2)",
      borderColor: "rgba(102,204,0,1)",
      pointRadius: 0,
      pointBackgroundColor:  "rgba(102,204,0,1)",
      pointBorderColor: "rgba(255,255,255,0.8)",
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(102,204,0,1)",
      pointHitRadius: 50,
      pointBorderWidth: 2,
      data: [4.4,4.4,4.2,4.1,4.1,3.8,3.8,4.0,3.7,3.7,3.5,3.9,12.1],
    },
    {
      label: "Women",
      lineTension: false,
      fill: false,
      backgroundColor: "rgba(255,51,0,0.2)",
      borderColor: "rgba(255,51,0,1)",
      pointRadius: 0,
      pointBackgroundColor: "rgba(255,51,0,1)",
      pointBorderColor: "rgba(255,255,255,0.8)",
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(255,51,0,1)",
      pointHitRadius: 50,
      pointBorderWidth: 2,
      data: [4.4,4.3,4.1,4.0,3.8,3.8,3.8,3.8,3.6,3.6,3.5,3.8,14.1],
    }
    ],
  },
  options: {
    maintainAspectRatio: false,
    scales: {
      xAxes: [{
        time: {
          unit: 'date'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 30
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          max: 20,
          maxTicksLimit: 5
        },
        gridLines: {
          color: "rgba(0, 0, 0, 0.1)",
        }
      }],
    },
    legend: {
      display: true
    }
  }
});
