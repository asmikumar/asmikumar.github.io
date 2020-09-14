// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Pie Chart Example
var ctx = document.getElementById("education_health_race").getContext('2d');
var education_health_race = new Chart(ctx, {
  type: 'pie',
  data: {
    labels: ["White", "Black", "Hispanic"],
    datasets: [{
      backgroundColor: ["rgba(0,153,255,1)", "rgba(102,102,204,1)", "rgba(255,204,0,1)"],
      data: [75.3, 15.1, 13.5]
    }]
  },
  options: {
    maintainAspectRatio: false
  }
});