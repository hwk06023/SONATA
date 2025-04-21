---
layout: page
title: Search
permalink: /search
nav_exclude: true
---

# Search

<div class="search">
  <div class="search-input-wrap">
    <input type="text" id="search-input" class="search-input" tabindex="0" placeholder="Search documentation" aria-label="Search documentation" autocomplete="off">
    <label for="search-input" class="search-label"><i class="fa fa-search"></i></label>
  </div>
  <div id="search-results" class="search-results"></div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  
  searchInput.addEventListener('input', function() {
    const query = this.value.trim();
    
    if (query.length < 2) {
      searchResults.innerHTML = '';
      return;
    }
    
    if (window.jtd && window.jtd.search) {
      const results = window.jtd.search.search(query);
      displayResults(results);
    }
  });
  
  function displayResults(results) {
    if (results.length === 0) {
      searchResults.innerHTML = '<p>No results found</p>';
      return;
    }
    
    let resultsHtml = '<ul>';
    results.forEach(function(result) {
      resultsHtml += `
        <li>
          <a href="${result.url}" class="search-result-link">
            <div class="search-result-title">${result.title}</div>
            <div class="search-result-preview">${result.content.substring(0, 150)}...</div>
          </a>
        </li>
      `;
    });
    resultsHtml += '</ul>';
    
    searchResults.innerHTML = resultsHtml;
  }
});
</script>

<style>
.search-input-wrap {
  position: relative;
  width: 100%;
  max-width: 600px;
  margin: 0 auto 1rem;
}

.search-input {
  width: 100%;
  padding: 0.75rem 1rem 0.75rem 2.5rem;
  font-size: 16px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background-color: var(--background-color);
  color: var(--text-color);
}

.search-label {
  position: absolute;
  left: 1rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-muted);
}

.search-results {
  margin-top: 1.5rem;
}

.search-results ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.search-results li {
  margin-bottom: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border-color);
}

.search-results li:last-child {
  border-bottom: none;
}

.search-result-link {
  display: block;
  color: var(--link-color);
  text-decoration: none;
}

.search-result-title {
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.search-result-preview {
  color: var(--text-color);
  font-size: 14px;
}
</style> 