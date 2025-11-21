document.addEventListener('DOMContentLoaded', () => {
  const modelSelect = document.getElementById('model-select');
  const roleSelect = document.getElementById('role-select');
  const taskFiltersGroup = document.getElementById('task-filter-group');
  const taskFiltersContainer = document.getElementById('task-filters');
  const responsesContainer = document.getElementById('responses-container');
  const configDisplay = document.getElementById('config-display');
  const systemPromptContent = document.getElementById('system-prompt-content');
  const taskPromptsList = document.getElementById('task-prompts-list');
  const loadingIndicator = document.getElementById('loading-indicator');

  let manifest = null;
  let currentModelData = null;
  let currentRoleData = null;
  let activeTaskFilters = new Set();

  // Initialize
  fetchDataManifest();

  // Event Listeners
  modelSelect.addEventListener('change', handleModelChange);
  roleSelect.addEventListener('change', handleRoleChange);

  async function fetchDataManifest() {
    showLoading(true);
    try {
      const response = await fetch('data_manifest.json');
      manifest = await response.json();
      populateModelSelect();
    } catch (error) {
      console.error('Error fetching manifest:', error);
      responsesContainer.innerHTML = '<div class="empty-state">Error loading data. Please try again.</div>';
    } finally {
      showLoading(false);
    }
  }

  function populateModelSelect() {
    modelSelect.innerHTML = '<option value="" disabled selected>Select a model...</option>';
    manifest.models.forEach(model => {
      const option = document.createElement('option');
      option.value = model.name;
      option.textContent = model.name;
      modelSelect.appendChild(option);
    });
  }

  function handleModelChange() {
    const modelName = modelSelect.value;
    currentModelData = manifest.models.find(m => m.name === modelName);

    // Reset UI
    roleSelect.innerHTML = '<option value="" disabled selected>Select a role...</option>';
    roleSelect.disabled = false;
    taskFiltersGroup.style.display = 'none';
    taskFiltersContainer.innerHTML = '';
    responsesContainer.innerHTML = '<div class="empty-state">Select a role to view responses.</div>';

    // Populate Roles
    if (currentModelData && currentModelData.roles) {
      currentModelData.roles.forEach(role => {
        const option = document.createElement('option');
        option.value = role;
        option.textContent = role;
        roleSelect.appendChild(option);
      });
    }

    // Display Config
    displayConfig(currentModelData.config);
  }

  async function handleRoleChange() {
    const roleName = roleSelect.value;
    const modelName = currentModelData.name;

    if (!roleName) return;

    showLoading(true);
    try {
      const response = await fetch(`data/${modelName}/${roleName}.json`);
      currentRoleData = await response.json();

      // Setup Task Filters
      setupTaskFilters(currentRoleData);

      // Render Responses
      renderResponses();
    } catch (error) {
      console.error('Error fetching role data:', error);
      responsesContainer.innerHTML = '<div class="empty-state">Error loading role data.</div>';
    } finally {
      showLoading(false);
    }
  }

  function displayConfig(config) {
    if (!config) {
      configDisplay.classList.add('hidden');
      return;
    }

    configDisplay.classList.remove('hidden');

    // System Prompt
    if (config.SYSTEM_PROMPT) {
      systemPromptContent.textContent = config.SYSTEM_PROMPT;
    } else {
      systemPromptContent.textContent = "No system prompt available.";
    }

    // Task Prompts
    taskPromptsList.innerHTML = '';
    if (config.TASK_PROMPTS) {
      for (const [key, value] of Object.entries(config.TASK_PROMPTS)) {
        const div = document.createElement('div');
        div.className = 'task-prompt-item';
        div.innerHTML = `<span class="task-prompt-key">${key}:</span> "${value}"`;
        taskPromptsList.appendChild(div);
      }
    } else {
      taskPromptsList.innerHTML = '<div>No task prompts available.</div>';
    }
  }

  function setupTaskFilters(data) {
    const tasks = new Set(data.map(item => item.task_name));
    activeTaskFilters = new Set(tasks); // All active by default

    taskFiltersContainer.innerHTML = '';
    taskFiltersGroup.style.display = 'block';

    // "Select All" / "Deselect All" logic could be added, but for now simple checkboxes

    tasks.forEach(task => {
      const label = document.createElement('label');
      label.className = 'checkbox-item';

      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = task;
      input.checked = true;
      input.addEventListener('change', (e) => {
        if (e.target.checked) {
          activeTaskFilters.add(task);
        } else {
          activeTaskFilters.delete(task);
        }
        renderResponses();
      });

      label.appendChild(input);
      label.appendChild(document.createTextNode(task));
      taskFiltersContainer.appendChild(label);
    });
  }

  function renderResponses() {
    responsesContainer.innerHTML = '';

    if (!currentRoleData || currentRoleData.length === 0) {
      responsesContainer.innerHTML = '<div class="empty-state">No responses found.</div>';
      return;
    }

    const filteredData = currentRoleData.filter(item => activeTaskFilters.has(item.task_name));

    if (filteredData.length === 0) {
      responsesContainer.innerHTML = '<div class="empty-state">No responses match the selected filters.</div>';
      return;
    }

    filteredData.forEach(item => {
      const card = document.createElement('div');
      card.className = 'response-card';

      const header = document.createElement('div');
      header.className = 'response-header';

      const meta = document.createElement('div');
      meta.className = 'response-meta';

      // Badges
      const roleBadge = createBadge(item.role_name, 'role');
      const taskBadge = createBadge(item.task_name, 'task');
      const sampleBadge = createBadge(`Sample ${item.sample_idx}`, 'sample');

      meta.appendChild(roleBadge);
      meta.appendChild(taskBadge);
      meta.appendChild(sampleBadge);

      // Add rating badges if available
      if (item.ratings) {
        if (item.ratings.assistant_refusal) {
          meta.appendChild(createBadge('Assistant Refusal', 'rating-negative'));
        }
        if (item.ratings.role_refusal) {
          meta.appendChild(createBadge('Role Refusal', 'rating-negative'));
        }
        if (item.ratings.identify_as_assistant) {
          meta.appendChild(createBadge('Identifies as Assistant', 'rating-negative'));
        }
        if (item.ratings.deny_internal_experience) {
          meta.appendChild(createBadge('Denies Internal Experience', 'rating-negative'));
        }
      }

      header.appendChild(meta);
      card.appendChild(header);

      const body = document.createElement('div');
      body.className = 'response-body';
      // Use marked to render markdown
      body.innerHTML = marked.parse(item.response || '');

      card.appendChild(body);
      responsesContainer.appendChild(card);
    });
  }

  function createBadge(text, type) {
    const span = document.createElement('span');
    span.className = `badge badge-${type}`;
    span.textContent = text;
    return span;
  }

  function showLoading(isLoading) {
    if (isLoading) {
      loadingIndicator.classList.remove('hidden');
    } else {
      loadingIndicator.classList.add('hidden');
    }
  }
});
