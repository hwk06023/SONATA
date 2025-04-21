(function() {
  'use strict';

  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  const searchContainer = document.querySelector('.search');
  
  if (!searchInput || !searchResults) return;
  
  let searchIndex;
  let searchData;
  let searchTimeout;
  
  const debounce = (fn, time) => {
    return (...args) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => fn(...args), time);
    };
  };
  
  const initSearch = async () => {
    const response = await fetch('/SONATA/assets/js/search-data.json');
    searchData = await response.json();
    
    searchIndex = lunr(function() {
      this.ref('id');
      this.field('title', { boost: 10 });
      this.field('content');
      this.metadataWhitelist = ['position'];
      
      searchData.forEach((doc, idx) => {
        this.add({
          id: idx,
          title: doc.title,
          content: doc.content
        });
      });
    });
    
    searchInput.addEventListener('input', debounce(performSearch, 200));
    searchInput.addEventListener('keydown', handleKeyDown);
  };
  
  const performSearch = () => {
    const query = searchInput.value.trim();
    
    if (query.length < 2) {
      hideResults();
      return;
    }
    
    try {
      const results = searchIndex.search(query);
      displayResults(results);
    } catch (e) {
      console.error('Search error:', e);
      searchResults.innerHTML = '<p>An error occurred with the search. Please try again with a different query.</p>';
    }
  };
  
  const displayResults = (results) => {
    if (results.length === 0) {
      searchResults.innerHTML = '<p>No results found</p>';
      return;
    }
    
    let resultsHtml = '<ul>';
    
    results.slice(0, 10).forEach(r => {
      const item = searchData[r.ref];
      const title = item.title;
      const url = item.url;
      let preview = item.content.substring(0, 150);
      
      if (preview.length === 150) {
        preview += '...';
      }
      
      resultsHtml += `
        <li>
          <a href="${url}" class="search-result-link">
            <div class="search-result-title">${title}</div>
            <div class="search-result-preview">${preview}</div>
          </a>
        </li>
      `;
    });
    
    resultsHtml += '</ul>';
    searchResults.innerHTML = resultsHtml;
    searchContainer.classList.add('active');
  };
  
  const hideResults = () => {
    searchResults.innerHTML = '';
    searchContainer.classList.remove('active');
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      hideResults();
      searchInput.value = '';
    }
  };
  
  // Close search results when clicking outside
  document.addEventListener('click', (e) => {
    if (!searchContainer.contains(e.target)) {
      hideResults();
    }
  });
  
  // Initialize search when Lunr is available
  if (typeof lunr !== 'undefined') {
    initSearch();
  } else {
    console.warn('Lunr.js is not loaded - search functionality disabled');
  }
})(); 