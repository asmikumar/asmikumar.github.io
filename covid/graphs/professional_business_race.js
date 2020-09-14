// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Pie Chart Example
var ctx = document.getElementById("professional_business_race").getContext('2d');
var professional_business_race = new Chart(ctx, {
  type: 'pie',
  data: {
    labels: ["White", "Black", "Hispanic"],
    datasets: [{
      backgroundColor: ["rgba(0,153,255,1)", "rgba(102,102,204,1)", "rgba(255,204,0,1)"],
      data: [77.4, 10.0, 16.0]
    }]
  },
  options: {
    maintainAspectRatio: false
  }
});