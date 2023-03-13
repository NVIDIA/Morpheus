$(document).ready(function() { 
         
    $("#submit").click(function() {
      convertToJson();
    });
 
  // Function to convert inputs-container and child data to JSON
  function convertToJson() {
  var inputsContainer = $('#inputs-container');
  var inputs = inputsContainer.find('.input');
  var jsonOutput = {};
  jsonOutput.inputs = [];
 
  inputs.each(function(index) {
     var input = $(this);
     var dataType = input.find('select[name="data-type"]').val();
     var metadataContainer = input.find('.metadata-container');
     var metadata = metadataContainer.find('.metadata');
     var metadataJson = {};
 
     metadata.each(function(index) {
        var metadataItem = $(this);
        var key = metadataItem.find('input[name="metadata-key"]').val();
        var dataType = metadataItem.find('select[name="metadata-data-type"]').val();
        var value = metadataItem.find('input[name="metadata-value"]').val();
 
        if (dataType === "text-array")
              value = value.split(",");
 
        metadataJson[key] = value;
     });
 
     var tasksContainer = input.find('.tasks-container');
     var tasks = tasksContainer.find('.task');
     var tasksJson = [];
 
     tasks.each(function(index) {
        var task = $(this);
        var taskType = task.find('select[name="task-type"]').val();
        var propertiesContainer = task.find('.properties-container');
        var properties = propertiesContainer.find('.property');
        var propertiesJson = {};
 
        properties.each(function(index) {
        var property = $(this);
        var key = property.find('input[name="property-key"]').val();
        var dataType = property.find('select[name="property-data-type"]').val();
        var value = property.find('input[name="property-value"]').val();
 
        if (dataType === "text-array")
           value = value.split(",");
 
        propertiesJson[key] = value;
        });
 
        tasksJson.push({ "type": taskType, "properties": propertiesJson });
     });
 
     metadataJson['data_type'] = dataType
     var inputJson = { "metadata": metadataJson, "tasks": tasksJson };
     jsonOutput.inputs.push(inputJson);
  });
 
  var jsonString = JSON.stringify(jsonOutput, null, 2);
  $('#control-messages-json').val(jsonString);
  }
 
 
 
    // Add new input button functionality
    $("#add-input-btn").click(function() {
      var inputHtml = `
      <div class="input">
        <label>Data Type:</label>
        <select name="data-type">
          <option value="payload">Payload</option>
          <option value="streaming">Streaming</option>
        </select>
        <button type="button" class="add-metadata-btn">Add Metadata</button>
        <div class="metadata-container">
          <!-- Metadata will be dynamically added here -->
        </div>
        <div class="tasks-container">
          <!-- Tasks will be dynamically added here -->
        </div>
        <button type="button" class="add-task-btn">Add Task</button>
        <button type="button" class="remove-input-btn">Remove Control Message</button>
      </div>`;
      $("#inputs-container").append(inputHtml);
    });
  
    // Add new task button functionality using event delegation
    $("#inputs-container").on("click", ".add-task-btn", function() {
      var taskHtml = `
      <div class="task">
        <label>Type:</label>
        <select name="task-type">
          <option value="load">Load</option>
          <option value="inference">Inference</option>
          <option value="training">Training</option>
        </select>
        <button type="button" class="add-property-btn">Add Property</button>
        <div class="properties-container">
          <!-- Properties will be dynamically added here -->
        </div>
        <button type="button" class="remove-task-btn">Remove Task</button>
      </div>`;
      $(this).parent().find(".tasks-container").append(taskHtml);
    });
  
    // Add new property button functionality
    $("#inputs-container").on("click", ".add-property-btn", function() {
      var propertyHtml = `
      <div class="property">
        <label>Key:</label>
        <input type="text" name="property-key">
        <label>Data Type:</label>
        <select name="property-data-type">
          <option value="text">Text</option>
          <option value="number">Number</option>
          <option value="boolean">Boolean</option>
          <option value="date">Date</option>
          <option value="text-array">Array</option>
        </select>
        <label>Value:</label>
        <input type="text" name="property-value">
        <button type="button" class="remove-property-btn">Remove</button>
      </div>`;
      $(this).siblings(".properties-container").append(propertyHtml);
    });
  
    $("#inputs-container").on("click", ".add-metadata-btn", function() {
      var metadataHtml = `
      <div class="metadata">
        <label>Key:</label>
        <input type="text" name="metadata-key">
        <label>Data Type:</label>
        <select name="metadata-data-type">
          <option value="text">Text</option>
          <option value="number">Number</option>
          <option value="boolean">Boolean</option>
          <option value="date">Date</option>
          <option value="text-array">Array</option>
        </select>
        <label>Value:</label>
        <input type="text" name="metadata-value">
        <button type="button" class="remove-metadata-btn">Remove</button>
      </div>`;
      $(this).siblings(".metadata-container").append(metadataHtml);
    });
    
    // Remove input button functionality using event delegation
    $("#inputs-container").on("click", ".remove-input-btn", function() {
       $(this).parent().remove();
     });
 
     // Remove task button functionality using event delegation
     $("#inputs-container").on("click", ".remove-task-btn", function() {
       $(this).parent().remove();
     });
 
     // Remove property button functionality using event delegation
     $("#inputs-container").on("click", ".remove-property-btn", function() {
       $(this).parent().remove();
     });
 
     // Remove metadata button functionality using event delegation
     $("#inputs-container").on("click", ".remove-metadata-btn", function() {
       $(this).parent().remove();
     });
 
  });